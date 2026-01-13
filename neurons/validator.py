"""Tournament validator - evaluates miner submissions and sets weights."""

import argparse
import asyncio
import gc
import logging
import os
import statistics
import time

import bittensor as bt
import torch

from tournament.chain.weights import WeightSetter
from tournament.config import get_config, get_hparams
from tournament.core.protocols import SubmissionStatus
from tournament.payment.verifier import PaymentVerifier
from tournament.pipeline.validator import CodeValidator
from tournament.sandbox.manager import SandboxManager
from tournament.schemas import BenchmarkConfig
from tournament.storage.database import Database, get_database
from tournament.storage.models import EvaluationModel
from tournament.storage.r2 import get_r2_storage
from tournament.verification import (
    ReferenceExecutor,
    SandboxVerifier,
    VerificationConfig,
)

from .base_node import BaseNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class Validator(BaseNode):
    """Tournament validator node.

    Responsibilities:
    1. Validate pending submissions (syntax, required functions)
    2. Evaluate validated submissions in sandbox
    3. Calculate and set weights (winner-takes-all)
    """

    def __init__(
        self,
        wallet: bt.wallet | None = None,
        burn_hotkey: str | None = None,
        burn_enabled: bool = False,
        skip_blockchain_check: bool = False,
    ):
        super().__init__(wallet=wallet)

        self.burn_hotkey = burn_hotkey
        self.burn_enabled = burn_enabled
        self.skip_blockchain_check = skip_blockchain_check

        # Components (initialized in start)
        self.db: Database | None = None
        self.sandbox: SandboxManager | None = None
        self.verifier: SandboxVerifier | None = None
        self.code_validator: CodeValidator | None = None
        self.weight_setter: WeightSetter | None = None
        self.payment_verifier: PaymentVerifier | None = None

        # Timing
        self.last_weight_set_time: float = 0
        self.last_sync_time: float = 0

    async def initialize(self) -> None:
        """Initialize components."""
        config = get_config()
        hparams = get_hparams()

        # Database
        self.db = await get_database()

        # Sandbox (uses test.pt for evaluation - hidden from miners)
        self.sandbox = SandboxManager(
            benchmark_model_path=config.benchmark_model_path,
            benchmark_data_path=config.benchmark_data_path,  # Points to test.pt for validators
        )
        await self.sandbox.initialize()

        # Verification configuration (read from hparams.json)
        verification_dict = (
            hparams.verification.model_dump() 
            if hasattr(hparams, "verification") and hparams.verification 
            else {}
        )
        verification_config = VerificationConfig.from_dict(verification_dict)

        # Reference executor (uses test.pt for evaluation)
        benchmark_config = BenchmarkConfig(
            model_path=config.benchmark_model_path,
            data_path=config.benchmark_data_path,  # Points to test.pt for validators
            sequence_length=hparams.benchmark_sequence_length,
            batch_size=hparams.benchmark_batch_size,
            num_steps=hparams.eval_steps,
        )
        reference_executor = ReferenceExecutor(
            model_path=config.benchmark_model_path,
            data_path=config.benchmark_data_path,  # Points to test.pt for validators
            config=benchmark_config,
        )

        # Verifier (combines sandbox + reference for verification)
        self.verifier = SandboxVerifier(
            sandbox_manager=self.sandbox,
            reference_executor=reference_executor,
            config=verification_config,
        )

        # Code validator
        self.code_validator = CodeValidator()

        # Weight setter
        self.weight_setter = WeightSetter(
            chain=self.chain,
            database=self.db,
            burn_hotkey=self.burn_hotkey,
            burn_enabled=self.burn_enabled,
        )

        # Payment verifier (anti-spam)
        self.payment_verifier = PaymentVerifier(
            recipient_address=self.wallet.hotkey.ss58_address,
            subtensor=self.chain.subtensor,
        )

    async def start(self) -> None:
        """Start the validator."""
        await self.initialize()
        await super().start()

    async def run_step(self) -> None:
        """Run one iteration of the validator loop."""
        logger.info("🔄 Starting validation loop iteration...")
        
        # 1. Process pending submissions (validation)
        logger.info("Step 1: Processing pending submissions...")
        await self.process_pending_submissions()

        # 2. Evaluate submissions ready for evaluation
        logger.info("Step 2: Evaluating submissions...")
        await self.evaluate_submissions()

        # 3. Set weights periodically (skip on localnet - not supported)
        logger.info("Step 3: Checking weight setting...")
        if self.config.subtensor_network != "local":
            await self.maybe_set_weights()
        else:
            logger.info("Skipping weight setting (localnet mode)")

        # 4. Sync metagraph periodically (every 5 minutes)
        logger.info("Step 4: Checking metagraph sync...")
        await self.maybe_sync()
        
        # 5. Memory cleanup to prevent OOM
        self._cleanup_memory()

        # Sleep before next iteration
        logger.info("✅ Loop iteration complete. Sleeping 10s...")
        await asyncio.sleep(10)
    
    def _cleanup_memory(self):
        """Clean up GPU memory after each loop iteration.
        
        Prevents OOM errors in long-running validators by:
        1. Clearing PyTorch's memory cache
        2. Running garbage collection
        3. Periodic aggressive cleanup every 10 iterations
        """
        if not hasattr(self, '_loop_count'):
            self._loop_count = 0
        
        self._loop_count += 1
        
        # Light cleanup every iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Aggressive cleanup every 10 iterations
        if self._loop_count % 10 == 0:
            logger.info(f"🧹 Aggressive memory cleanup (iteration {self._loop_count})...")
            
            if torch.cuda.is_available():
                # Clear all caches
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Force garbage collection
                gc.collect()
                
                # Log memory status
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    logger.info(f"   GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                    
                    if reserved > allocated * 2:
                        logger.warning(f"   GPU {i}: High fragmentation ({reserved - allocated:.2f}GB wasted)")
            
            gc.collect()

    async def maybe_sync(self) -> None:
        """Sync metagraph periodically."""
        now = time.time()
        if now - self.last_sync_time >= 300:  # Every 5 minutes
            await self.sync()
            self.last_sync_time = now

    async def process_pending_submissions(self) -> None:
        """Validate pending submissions (including payment verification)."""
        pending = await self.db.get_pending_submissions()

        for submission in pending:
            logger.info(f"Validating submission {submission.submission_id}")

            # Update status to validating
            await self.db.update_submission_status(
                submission.submission_id,
                SubmissionStatus.VALIDATING,
            )

            # Step 1: Verify payment (if payment info provided)
            if submission.payment_block_hash and not submission.payment_verified:
                logger.info(f"Verifying payment for submission {submission.submission_id}")

                payment_valid, payment_error = await self.payment_verifier.verify_payment(
                    block_hash=submission.payment_block_hash,
                    extrinsic_index=submission.payment_extrinsic_index,
                    miner_coldkey=submission.miner_hotkey,  # Assuming hotkey used for payment
                    expected_amount_rao=submission.payment_amount_rao,
                )

                if not payment_valid:
                    await self.db.update_submission_status(
                        submission.submission_id,
                        SubmissionStatus.FAILED_VALIDATION,
                        error_message=f"Payment verification failed: {payment_error}",
                    )
                    logger.warning(
                        f"Submission {submission.submission_id} failed payment verification: {payment_error}"
                    )
                    continue

                # Mark payment as verified
                submission.payment_verified = True
                logger.info(f"Payment verified for submission {submission.submission_id}")

            # Step 2: Download code from R2
            # TODO: Implement R2 download
            # For now, assume code is available locally
            # code = await download_from_r2(submission.bucket_path)

            # Step 3: Validate code
            # result = self.code_validator.validate(code)
            # For now, just mark as evaluating
            result_valid = True

            if result_valid:
                await self.db.update_submission_status(
                    submission.submission_id,
                    SubmissionStatus.EVALUATING,
                )
                logger.info(f"Submission {submission.submission_id} passed validation")
            else:
                await self.db.update_submission_status(
                    submission.submission_id,
                    SubmissionStatus.FAILED_VALIDATION,
                    error_message="Validation failed",  # result.errors
                )
                logger.warning(f"Submission {submission.submission_id} failed validation")

    async def evaluate_submissions(self) -> None:
        """Evaluate submissions that are ready with verification.
        
        FAIRNESS: Each submission is evaluated 3 times (configurable).
        The final score is the MEDIAN of successful evaluations.
        
        Why median?
        - Protects against random GPU hiccups (one bad run doesn't tank your score)
        - More fair than average (outliers are ignored)
        - Example: [1200, 1250, 50] → median=1200 (fair), average=833 (unfair)
        """
        hparams = get_hparams()
        evaluating = await self.db.get_evaluating_submissions()
        
        # Number of evaluation runs per submission (from hparams.json)
        num_runs = hparams.evaluation_runs

        for submission in evaluating:
            # Check if we've already evaluated this submission enough times
            existing_evals = await self.db.get_evaluations(submission.submission_id)
            my_evals = [e for e in existing_evals if e.evaluator_hotkey == self.hotkey]

            if len(my_evals) >= num_runs:
                # Already did all evaluation runs for this validator
                continue

            runs_remaining = num_runs - len(my_evals)
            logger.info(f"Evaluating submission {submission.submission_id} (run {len(my_evals)+1}/{num_runs})")

            # Download code from R2
            r2_storage = get_r2_storage()
            code_path = f"/tmp/submissions/{submission.submission_id}/train.py"

            logger.info(f"Downloading code from storage: {submission.bucket_path}")
            download_success = await r2_storage.download_code(
                submission.bucket_path, code_path
            )

            if not download_success:
                logger.error(f"Failed to download code for submission {submission.submission_id}")
                # Save failed evaluation
                evaluation = EvaluationModel(
                    submission_id=submission.submission_id,
                    evaluator_hotkey=self.hotkey,
                    tokens_per_second=0.0,
                    total_tokens=0,
                    wall_time_seconds=0.0,
                    success=False,
                    error="Failed to download code from storage",
                )
                await self.db.save_evaluation(evaluation)
                continue

            logger.info(f"Code downloaded successfully: {code_path}")

            # Run multiple evaluations for fairness
            for run_idx in range(runs_remaining):
                current_run = len(my_evals) + run_idx + 1
                logger.info(f"🔄 Evaluation run {current_run}/{num_runs}")
                
                # Run verification and benchmarking
                result = await self.verifier.verify_and_benchmark(code_path)

                if result.success:
                    logger.info(
                        f"✅ Run {current_run} PASSED for {submission.submission_id}\n"
                        f"  TPS: {result.tokens_per_second:,.2f}\n"
                        f"  Tokens: {result.total_tokens:,}\n"
                        f"  Time: {result.wall_time_seconds:.2f}s"
                    )
                else:
                    logger.warning(
                        f"❌ Run {current_run} FAILED for {submission.submission_id}\n"
                        f"  Error type: {result.error_type}\n"
                        f"  Message: {result.error_message}"
                    )

                # Save evaluation result
                evaluation = EvaluationModel(
                    submission_id=submission.submission_id,
                    evaluator_hotkey=self.hotkey,
                    tokens_per_second=result.tokens_per_second,
                    total_tokens=result.total_tokens,
                    wall_time_seconds=result.wall_time_seconds,
                    success=result.success,
                    error=result.error_message,
                )
                await self.db.save_evaluation(evaluation)
                
                # Clean up memory between runs
                self._cleanup_memory()

            # Check if submission has enough evaluations from all validators
            num_evals = await self.db.count_evaluations(submission.submission_id)
            required_evals = hparams.num_evals_per_submission * num_runs
            
            if num_evals >= required_evals:
                # Calculate final score using MEDIAN (fair - ignores outliers)
                all_evals = await self.db.get_evaluations(submission.submission_id)
                successful_evals = [e for e in all_evals if e.success]

                if successful_evals:
                    tps_scores = [e.tokens_per_second for e in successful_evals]
                    
                    # Use median for fairness (protects against random bad runs)
                    median_tps = statistics.median(tps_scores)
                    avg_tps = statistics.mean(tps_scores)
                    
                    logger.info(
                        f"📊 Final scoring for {submission.submission_id}:\n"
                        f"   Runs: {len(tps_scores)} successful\n"
                        f"   Scores: {[f'{s:.1f}' for s in sorted(tps_scores)]}\n"
                        f"   Median: {median_tps:,.2f} TPS (FINAL)\n"
                        f"   Average: {avg_tps:,.2f} TPS (for reference)"
                    )
                    
                    # Use median as final score
                    await self.db.update_submission_score(submission.submission_id, median_tps)
                    logger.info(
                        f"✅ Submission {submission.submission_id} finished with score {median_tps:,.2f} TPS (median)"
                    )
                else:
                    # All evaluations failed verification
                    await self.db.update_submission_status(
                        submission.submission_id,
                        SubmissionStatus.FAILED_VALIDATION,
                        error_message="All evaluations failed verification",
                    )
                    logger.warning(
                        f"Submission {submission.submission_id} failed: no successful evaluations"
                    )

    async def maybe_set_weights(self) -> None:
        """Set weights if enough time has passed."""
        hparams = get_hparams()
        now = time.time()

        if now - self.last_weight_set_time < hparams.set_weights_interval_seconds:
            return

        logger.info("Setting weights...")
        success, message = await self.weight_setter.set_weights()

        if success:
            self.last_weight_set_time = now
            logger.info(f"Weights set: {message}")
        else:
            logger.warning(f"Failed to set weights: {message}")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await super().cleanup()
        if self.sandbox:
            await self.sandbox.cleanup()
        if self.db:
            await self.db.close()


def main():
    # Configure PyTorch memory management to prevent OOM
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    logger.info("🧠 Memory management configured: expandable_segments=True")
    
    parser = argparse.ArgumentParser(description="Tournament Validator")
    parser.add_argument(
        "--wallet.name",
        dest="wallet_name",
        type=str,
        default="default",
        help="Wallet name",
    )
    parser.add_argument(
        "--wallet.hotkey",
        dest="wallet_hotkey",
        type=str,
        default="default",
        help="Wallet hotkey",
    )
    parser.add_argument(
        "--burn-hotkey",
        type=str,
        default=None,
        help="Hotkey to burn emissions to when no winner",
    )
    parser.add_argument(
        "--burn-enabled",
        action="store_true",
        help="Enable burn mode (all emissions to burn hotkey)",
    )
    parser.add_argument(
        "--skip-blockchain-check",
        action="store_true",
        help="Skip blockchain registration check (for local testing)",
    )

    args = parser.parse_args()

    # Initialize wallet
    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    # Create and run validator
    validator = Validator(
        wallet=wallet,
        burn_hotkey=args.burn_hotkey,
        burn_enabled=args.burn_enabled,
        skip_blockchain_check=args.skip_blockchain_check,
    )

    asyncio.run(validator.start())


if __name__ == "__main__":
    main()
