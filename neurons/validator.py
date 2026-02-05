"""Crusades validator - evaluates miner submissions and sets weights.

URL-Based Architecture:
1. Reads code URL commitments from blockchain (timelock decrypted)
2. Downloads miner's train.py from the committed URL
3. Evaluates via affinetes (Docker locally or Basilica remotely)
4. Sets weights based on TPS scores
"""

import argparse
import asyncio
import gc
import json
import logging
import os
import statistics
import time
import urllib.error
import urllib.request
from typing import Literal

import bittensor as bt
import torch

import crusades
from crusades.affinetes import AffinetesRunner
from crusades.chain.commitments import (
    CodeUrlInfo,
    CommitmentReader,
    MinerCommitment,
)
from crusades.chain.weights import WeightSetter
from crusades.config import get_config, get_hparams
from crusades.core.protocols import SubmissionStatus
from crusades.logging import setup_loki_logger
from crusades.storage.database import Database, get_database
from crusades.storage.models import EvaluationModel, SubmissionModel

from .base_node import BaseNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC | %(levelname)s | %(name)s | %(message)s",
)
# Force UTC for all log timestamps
logging.Formatter.converter = time.gmtime
logger = logging.getLogger(__name__)


class Validator(BaseNode):
    """Crusades validator node (URL-Based Architecture).

    Responsibilities:
    1. Read miner code URL commitments from blockchain
    2. Download train.py from URL and evaluate via affinetes
    3. Calculate and set weights (winner-takes-all)
    """

    def __init__(
        self,
        wallet: bt.wallet | None = None,
        skip_blockchain_check: bool = False,
        affinetes_mode: Literal["docker", "basilica"] = "docker",
    ):
        super().__init__(wallet=wallet, skip_blockchain_check=skip_blockchain_check)
        self.affinetes_mode = affinetes_mode

        # Components (initialized in start)
        self.db: Database | None = None
        self.weight_setter: WeightSetter | None = None
        self.commitment_reader: CommitmentReader | None = None
        self.affinetes_runner: AffinetesRunner | None = None

        # State
        self.last_processed_block: int = 0
        # Map URL -> (reveal_block, hotkey) to track first committer
        self.evaluated_code_urls: dict[str, tuple[int, str]] = {}

        # Timing
        self.last_weight_set_block: int = 0
        self.last_sync_time: float = 0

    async def initialize(self) -> None:
        """Initialize validator components."""
        global logger

        config = get_config()
        hparams = get_hparams()

        # Setup Loki logging for Grafana dashboard
        uid_str = str(self.uid) if self.uid is not None else "unknown"
        logger = setup_loki_logger(
            service="crusades-validator",
            uid=uid_str,
            version=crusades.__version__,
            environment=config.subtensor_network or "finney",
        )

        logger.info("Initializing validator (URL-Based Architecture)")

        # Database
        self.db = await get_database()

        # Weight setter and commitment reader
        if self.chain is not None:
            self.weight_setter = WeightSetter(
                chain=self.chain,
                database=self.db,
            )

            self.commitment_reader = CommitmentReader(
                subtensor=self.chain.subtensor,
                netuid=hparams.netuid,
                network=config.subtensor_network,
            )
            self.commitment_reader.sync()

            # Sync chain's metagraph first, then initialize weight block
            # This prevents spamming weight setting attempts on restart
            await self.chain.sync_metagraph()
            await self._init_weight_block_from_chain()
        else:
            logger.warning("Running without blockchain connection")
            logger.warning("Weight setting will be disabled")
            # Still create commitment reader - will connect lazily
            self.commitment_reader = CommitmentReader(
                subtensor=None,
                netuid=hparams.netuid,
                network=config.subtensor_network,
            )

        # Affinetes runner - all config comes from validated Pydantic models
        self.affinetes_runner = AffinetesRunner(
            mode=self.affinetes_mode,
            basilica_endpoint=os.getenv("BASILICA_ENDPOINT"),
            basilica_api_key=os.getenv("BASILICA_API_TOKEN"),
            # Docker config
            docker_gpu_devices=hparams.docker.gpu_devices,
            docker_memory_limit=hparams.docker.memory_limit,
            docker_shm_size=hparams.docker.shm_size,
            # Benchmark config
            model_url=hparams.benchmark_model_name,
            data_url=hparams.benchmark_dataset_name,
            timeout=hparams.eval_timeout,
            max_loss_difference=hparams.verification.max_loss_difference,
            min_params_changed_ratio=hparams.verification.min_params_changed_ratio,
            # Gradient verification
            gradient_cosine_min=hparams.verification.gradient_cosine_min,
            gradient_norm_ratio_min=hparams.verification.gradient_norm_ratio_min,
            gradient_norm_ratio_max=hparams.verification.gradient_norm_ratio_max,
            # MFU calculation
            gpu_peak_tflops=hparams.mfu.gpu_peak_tflops,
            # Basilica config
            basilica_image=hparams.basilica.image,
            basilica_ttl_seconds=hparams.basilica.ttl_seconds,
            basilica_gpu_count=hparams.basilica.gpu_count,
            basilica_gpu_models=hparams.basilica.gpu_models,
            basilica_min_gpu_memory_gb=hparams.basilica.min_gpu_memory_gb,
            basilica_cpu=hparams.basilica.cpu,
            basilica_memory=hparams.basilica.memory,
        )

        # Load persisted state
        await self._load_state()

        logger.info(f"   Affinetes mode: {self.affinetes_mode}")
        logger.info(f"   Model: {hparams.benchmark_model_name}")
        logger.info(f"   Dataset: {hparams.benchmark_dataset_name}")
        if self.affinetes_mode == "docker":
            logger.info(f"   Docker GPU devices: {hparams.docker.gpu_devices}")
            logger.info(f"   Docker memory limit: {hparams.docker.memory_limit}")
            logger.info(f"   Docker shm size: {hparams.docker.shm_size}")
        elif self.affinetes_mode == "basilica":
            logger.info(f"   Basilica image: {hparams.basilica.image}")
            logger.info(f"   Basilica TTL: {hparams.basilica.ttl_seconds}s")
            logger.info(
                f"   Basilica GPU: {hparams.basilica.gpu_count}x {hparams.basilica.gpu_models}"
            )
            logger.info(f"   Basilica min GPU memory: {hparams.basilica.min_gpu_memory_gb}GB")

    async def start(self) -> None:
        """Start the validator."""
        await self.initialize()
        await super().start()

    async def run_step(self) -> None:
        """Run one iteration of the validator loop."""
        logger.info("Starting validation loop iteration...")

        # 1. Read blockchain commitments
        logger.info("Step 1: Reading blockchain commitments...")
        await self.process_blockchain_commitments()

        # 2. Evaluate via affinetes
        logger.info("Step 2: Evaluating via affinetes...")
        await self.evaluate_submissions()

        # 3. Set weights
        logger.info("Step 3: Checking weight setting...")
        await self.maybe_set_weights()

        # 4. Sync metagraph
        logger.info("Step 4: Checking metagraph sync...")
        await self.maybe_sync()

        # Memory cleanup
        self._cleanup_memory()

        logger.info("Loop iteration complete. Sleeping 10s...")
        await asyncio.sleep(10)

    async def process_blockchain_commitments(self) -> None:
        """Read and process new gist commitments from blockchain."""
        try:
            new_commitments = self.commitment_reader.get_new_commitments_since(
                self.last_processed_block
            )

            current_block = self.commitment_reader.get_current_block()

            if new_commitments:
                logger.info(f"Found {len(new_commitments)} new commitments")

                for commitment in new_commitments:
                    logger.info(
                        f"Processing commitment from UID {commitment.uid}, hotkey: {commitment.hotkey[:16]}..."
                    )
                    logger.info(f"   Has valid URL: {commitment.has_valid_code_url()}")

                    # Skip if no valid code URL
                    if not commitment.has_valid_code_url():
                        logger.warning(
                            f"Skipping commitment without valid code URL: UID {commitment.uid}"
                        )
                        continue

                    # Use code URL as unique identifier - first committer wins
                    code_url = commitment.code_url_info.url
                    if code_url in self.evaluated_code_urls:
                        first_block, first_hotkey = self.evaluated_code_urls[code_url]
                        if commitment.reveal_block >= first_block:
                            logger.info(
                                f"Skipping duplicate URL from UID {commitment.uid} "
                                f"(first committed at block {first_block} by {first_hotkey[:16]}...)"
                            )
                            continue
                        else:
                            # This commitment is earlier - it should have priority
                            # This shouldn't happen if we process in order, but log it
                            logger.warning(
                                f"Found earlier commitment for URL from UID {commitment.uid} "
                                f"at block {commitment.reveal_block} (previously saw block {first_block})"
                            )

                    await self._create_submission_from_commitment(commitment)
                    # Record this URL with the committer info
                    self.evaluated_code_urls[code_url] = (
                        commitment.reveal_block,
                        commitment.hotkey,
                    )

            self.last_processed_block = current_block
            await self._save_state()

        except Exception as e:
            logger.error(f"Error processing commitments: {e}")

    async def _create_submission_from_commitment(
        self,
        commitment: MinerCommitment,
    ) -> None:
        """Create a submission record from a blockchain commitment."""
        hparams = get_hparams()
        version = crusades.COMPETITION_VERSION
        submission_id = f"v{version}_commit_{commitment.reveal_block}_{commitment.uid}"

        try:
            existing = await self.db.get_submission(submission_id)
            if existing:
                logger.debug(f"Submission {submission_id} already exists")
                return
        except Exception as e:
            logger.error(f"Database error checking existing submission: {e}")

        # Rate limiting
        min_blocks = hparams.min_blocks_between_commits
        last_submission = await self.db.get_latest_submission_by_hotkey(commitment.hotkey)

        if last_submission:
            try:
                # Handle both old (commit_block_uid) and new (vN_commit_block_uid) formats
                parts = last_submission.submission_id.split("_")
                if parts[0].startswith("v"):
                    # New format: v3_commit_79639_1
                    last_block = int(parts[2])
                else:
                    # Old format: commit_79639_1
                    last_block = int(parts[1])
                blocks_since = commitment.reveal_block - last_block

                if blocks_since < min_blocks:
                    logger.warning(
                        f"Rate limit: {commitment.hotkey[:16]}... submitted too soon "
                        f"({blocks_since} blocks, min={min_blocks}). Skipping."
                    )
                    return
            except (IndexError, ValueError):
                pass

        submission = SubmissionModel(
            submission_id=submission_id,
            miner_hotkey=commitment.hotkey,
            miner_uid=commitment.uid,
            code_hash=commitment.code_url_info.url,  # Use code URL as identifier
            bucket_path=commitment.code_url_info.url,  # Store code URL
            status=SubmissionStatus.EVALUATING,
            payment_verified=True,
            spec_version=crusades.COMPETITION_VERSION,
        )

        try:
            await self.db.save_submission(submission)
            logger.info(f"✓ Created submission: {submission_id}")
            logger.info(f"   Code URL: {commitment.code_url_info.url[:60]}...")
            logger.info(f"   UID: {commitment.uid}")
            logger.info(f"   Hotkey: {commitment.hotkey[:16]}...")
        except Exception as e:
            logger.error(f"Failed to save submission: {e}")
            logger.exception("Traceback:")

    def _download_from_url(self, code_url: str) -> tuple[bool, str]:
        """Download train.py code from a URL with SSRF protection.

        Validates that:
        1. URL resolves to a non-private IP address (SSRF protection)
        2. Redirects don't lead to private IP addresses
        3. Response is a single Python file, not HTML/folder

        Args:
            code_url: The URL containing train.py code

        Returns:
            Tuple of (success, code_or_error)
        """
        # SSRF Protection: Validate URL before making request
        code_url_info = CodeUrlInfo(url=code_url)
        is_safe, validation_result = code_url_info.validate_url_security()

        if not is_safe:
            logger.warning(f"SSRF protection blocked URL: {validation_result}")
            return False, f"URL blocked for security: {validation_result}"

        logger.debug(f"URL validated, resolved to IP: {validation_result}")

        try:
            # Use a custom opener that validates redirect destinations
            class SSRFSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
                """Custom redirect handler that validates redirect destinations for SSRF."""

                def redirect_request(self, req, fp, code, msg, headers, newurl):
                    """Validate redirect destination before following."""
                    # Validate the redirect URL
                    redirect_info = CodeUrlInfo(url=newurl)
                    is_redirect_safe, redirect_result = redirect_info.validate_url_security()

                    if not is_redirect_safe:
                        logger.warning(
                            f"SSRF protection blocked redirect to: {newurl} - {redirect_result}"
                        )
                        raise urllib.error.URLError(
                            f"Redirect blocked for security: {redirect_result}"
                        )

                    logger.debug(f"Redirect validated: {newurl} -> {redirect_result}")
                    return super().redirect_request(req, fp, code, msg, headers, newurl)

            # Build opener with SSRF-safe redirect handler
            opener = urllib.request.build_opener(SSRFSafeRedirectHandler())

            req = urllib.request.Request(code_url, headers={"User-Agent": "templar-crusades"})
            with opener.open(req, timeout=30) as response:
                code = response.read().decode("utf-8")

                # Reject HTML (folder page, not raw file)
                if "<html" in code.lower()[:500] or "<!doctype html" in code.lower()[:500]:
                    return False, "URL returns HTML page, not a code file"

                # Reject JSON file listings
                if code.strip().startswith("{") and '"files"' in code[:500]:
                    return False, "URL returns JSON (file listing), not code"

                # Size limit (500KB)
                if len(code) > 500_000:
                    return False, f"File too large ({len(code)} bytes). Max 500KB"

                # Must contain inner_steps
                if "def inner_steps" not in code:
                    return False, "Code does not contain 'def inner_steps' function"

                return True, code

        except urllib.error.HTTPError as e:
            return False, f"HTTP error {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            return False, f"URL error: {e.reason}"
        except Exception as e:
            return False, f"Error downloading code: {e}"

    async def evaluate_submissions(self) -> None:
        """Evaluate submissions by downloading from code URL.

        Only evaluates submissions matching current competition version.
        """
        hparams = get_hparams()
        competition_version = crusades.COMPETITION_VERSION
        # Only evaluate submissions from current version
        evaluating = await self.db.get_evaluating_submissions(spec_version=competition_version)
        num_runs = hparams.evaluation_runs

        logger.info(
            f"Found {len(evaluating)} submissions in EVALUATING status (v{competition_version})"
        )

        for submission in evaluating:
            code_url = submission.bucket_path

            if not code_url or not code_url.startswith("http"):
                logger.error(f"Invalid code URL for {submission.submission_id}: {code_url}")
                continue

            existing_evals = await self.db.get_evaluations(submission.submission_id)
            my_evals = [e for e in existing_evals if e.evaluator_hotkey == self.hotkey]

            if len(my_evals) >= num_runs:
                continue

            runs_remaining = num_runs - len(my_evals)
            logger.info(f"Evaluating {submission.submission_id}")
            logger.info(f"   URL: {code_url[:60]}...")
            logger.info(f"   Runs: {len(my_evals) + 1}/{num_runs}")

            # Download code from URL
            success, code_or_error = self._download_from_url(code_url)

            if not success:
                logger.error(f"Failed to download code: {code_or_error}")
                await self.db.update_submission_status(
                    submission.submission_id,
                    SubmissionStatus.FAILED_EVALUATION,
                    error_message=f"Failed to download code: {code_or_error}",
                )
                continue

            miner_code = code_or_error
            logger.info(f"   Downloaded {len(miner_code)} bytes")

            # Run evaluations
            for run_idx in range(runs_remaining):
                current_run = len(my_evals) + run_idx + 1
                seed = f"{submission.miner_uid}:{current_run}:{int(time.time())}"

                logger.info(f"Evaluation run {current_run}/{num_runs} (seed: {seed})")

                result = await self.affinetes_runner.evaluate(
                    code=miner_code,  # Pass code directly
                    seed=seed,
                    steps=hparams.eval_steps,
                    batch_size=hparams.benchmark_batch_size,
                    sequence_length=hparams.benchmark_sequence_length,
                    data_samples=hparams.benchmark_data_samples,
                    task_id=current_run,
                )

                if result.success:
                    logger.info(
                        f"Run {current_run} PASSED: MFU={result.mfu:.2f}% TPS={result.tps:,.2f}"
                    )
                else:
                    logger.warning(f"Run {current_run} FAILED: {result.error}")

                evaluation = EvaluationModel(
                    submission_id=submission.submission_id,
                    evaluator_hotkey=self.hotkey,
                    mfu=result.mfu,  # MFU is primary metric
                    tokens_per_second=result.tps,
                    total_tokens=result.total_tokens,
                    wall_time_seconds=result.wall_time_seconds,
                    success=result.success,
                    error=result.error,
                )
                await self.db.save_evaluation(evaluation)
                self._cleanup_memory()

            # Persist state (URL already recorded during commitment processing)
            await self._save_state()

            # Store miner's code in database
            await self.db.update_submission_code(submission.submission_id, miner_code)
            logger.info(f"Stored code for {submission.submission_id} in database")

            # Finalize submission
            await self._finalize_submission(submission.submission_id, num_runs)

    async def _finalize_submission(self, submission_id: str, num_runs: int) -> None:
        """Calculate final score (MFU) and update submission status."""
        hparams = get_hparams()
        num_evals = await self.db.count_evaluations(submission_id)
        required_evals = num_runs

        logger.info(f"Finalizing {submission_id}: {num_evals}/{required_evals} evaluations")

        if num_evals >= required_evals:
            all_evals = await self.db.get_evaluations(submission_id)
            successful_evals = [e for e in all_evals if e.success]

            # Check minimum success rate
            success_rate = len(successful_evals) / len(all_evals) if all_evals else 0
            min_success_rate = getattr(hparams, "min_success_rate", 0.5)

            if success_rate < min_success_rate:
                await self.db.update_submission_status(
                    submission_id,
                    SubmissionStatus.FAILED_EVALUATION,
                    error_message=f"Success rate {success_rate:.1%} below minimum {min_success_rate:.0%}",
                )
                logger.warning(
                    f"Submission {submission_id} failed: success rate {success_rate:.1%} < {min_success_rate:.0%}"
                )
                return

            if successful_evals:
                # MFU is the primary metric now
                mfu_scores = [e.mfu for e in successful_evals]
                # Use median_low to always return an actual run value (not average of two)
                median_mfu = statistics.median_low(mfu_scores)

                logger.info(
                    f"Final score for {submission_id}:\n"
                    f"   Successful runs: {len(mfu_scores)}\n"
                    f"   MFU scores: {[f'{s:.2f}%' for s in sorted(mfu_scores)]}\n"
                    f"   Median MFU (low): {median_mfu:.2f}%"
                )

                await self.db.update_submission_score(submission_id, median_mfu)
                await self.db.update_submission_status(
                    submission_id,
                    SubmissionStatus.FINISHED,
                )
                logger.info(f"Submission {submission_id} FINISHED with MFU={median_mfu:.2f}%")
            else:
                # All evaluations failed - mark submission as failed with score 0
                await self.db.update_submission_score(submission_id, 0.0)
                await self.db.update_submission_status(
                    submission_id,
                    SubmissionStatus.FAILED_EVALUATION,
                    error_message="All evaluations failed",
                )
                logger.warning(f"Submission {submission_id} FAILED: all evaluations failed")

    def _cleanup_memory(self):
        """Clean up GPU memory."""
        if not hasattr(self, "_loop_count"):
            self._loop_count = 0

        self._loop_count += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self._loop_count % 10 == 0:
            logger.info(f"Memory cleanup (iteration {self._loop_count})")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

    async def maybe_sync(self) -> None:
        """Sync metagraph periodically."""
        now = time.time()
        if now - self.last_sync_time >= 300:
            await self.sync()
            if self.commitment_reader:
                self.commitment_reader.sync()
            self.last_sync_time = now

    async def _refresh_weight_block_from_chain(self) -> None:
        """Refresh last_weight_set_block from chain's metagraph.

        This ensures we have the latest last_update from chain before
        checking if we can set weights.
        """
        if self.chain is None:
            return

        try:
            # Sync metagraph to get latest state
            await self.chain.sync_metagraph()

            if self.chain.metagraph is None:
                return

            # Find our UID in the metagraph
            hotkey = self.chain.hotkey
            if hotkey not in self.chain.metagraph.hotkeys:
                return

            uid = self.chain.metagraph.hotkeys.index(hotkey)
            chain_last_update = int(self.chain.metagraph.last_update[uid])

            # Update if chain shows a more recent value
            if chain_last_update > self.last_weight_set_block:
                logger.debug(
                    f"Updating last_weight_set_block from chain: "
                    f"{self.last_weight_set_block} -> {chain_last_update}"
                )
                self.last_weight_set_block = chain_last_update
        except Exception as e:
            logger.debug(f"Could not refresh weight block from chain: {e}")

    async def maybe_set_weights(self) -> None:
        """Set weights if enough blocks have passed since last update."""
        if self.weight_setter is None:
            logger.debug("Skipping weight setting (test mode)")
            return

        if self.commitment_reader is None:
            logger.warning("Commitment reader not initialized - cannot set weights")
            return

        # Sync metagraph and refresh last_weight_set_block from chain
        # This ensures we don't attempt weight setting if chain shows recent update
        await self._refresh_weight_block_from_chain()

        hparams = get_hparams()
        current_block = self.commitment_reader.get_current_block()
        blocks_since_last = current_block - self.last_weight_set_block
        min_blocks = hparams.set_weights_interval_blocks

        if blocks_since_last <= min_blocks:
            # Don't log every time - only when we actually attempted
            return

        logger.info(
            f"Setting weights (block {current_block}, "
            f"{blocks_since_last} blocks since last update)..."
        )
        success, message = await self.weight_setter.set_weights()

        if success:
            self.last_weight_set_block = current_block
            logger.info(f"Weights set successfully at block {current_block}: {message}")
        else:
            # Provide detailed error info
            # Chain requires > min_blocks, so next allowed is last + min_blocks + 1
            next_allowed_block = self.last_weight_set_block + min_blocks + 1
            blocks_to_wait = max(0, next_allowed_block - current_block)
            logger.warning(
                f"Failed to set weights: {message}\n"
                f"  Current block: {current_block}\n"
                f"  Last successful: block {self.last_weight_set_block}\n"
                f"  Min interval: {min_blocks} blocks (chain requires >)\n"
                f"  Next allowed: block {next_allowed_block} ({blocks_to_wait} blocks to wait)"
            )

    async def _init_weight_block_from_chain(self) -> None:
        """Initialize last_weight_set_block from chain's metagraph.

        This prevents spamming weight setting attempts on validator restart
        by checking when the chain says we last set weights.
        """
        if self.chain is None or self.chain.metagraph is None:
            return

        try:
            # Find our UID in the metagraph
            hotkey = self.chain.hotkey
            if hotkey not in self.chain.metagraph.hotkeys:
                logger.warning("Validator hotkey not found in metagraph")
                return

            uid = self.chain.metagraph.hotkeys.index(hotkey)
            last_update = int(self.chain.metagraph.last_update[uid])

            if last_update > 0:
                self.last_weight_set_block = last_update
                logger.info(f"Initialized last_weight_set_block from chain: {last_update}")
        except Exception as e:
            logger.warning(f"Could not init weight block from chain: {e}")

    async def _load_state(self) -> None:
        """Load persisted validator state from database."""
        from crusades import COMPETITION_VERSION

        current_version = COMPETITION_VERSION

        try:
            # Check if version changed - if so, reset state for fresh competition
            stored_version_str = await self.db.get_validator_state("competition_version")
            stored_version = int(stored_version_str) if stored_version_str else 0

            if stored_version != current_version:
                # Get current block to start fresh from NOW (ignore old commitments)
                try:
                    current_block = self.chain.subtensor.get_current_block()
                except Exception as e:
                    # Cannot determine current block - defer version reset to avoid
                    # reprocessing all historical commitments from block 0
                    logger.warning(
                        f"Competition version changed ({stored_version} -> {current_version}), "
                        f"but cannot get current block: {e}. Deferring reset until chain is available."
                    )
                    return

                logger.info(
                    f"Competition version changed ({stored_version} -> {current_version}), "
                    f"starting fresh from block {current_block}"
                )
                # Keep last_processed_block at current block (ignore old commitments)
                self.last_processed_block = current_block
                # Clear evaluated URLs (allow same URLs in new version)
                self.evaluated_code_urls = {}
                # Save new version
                await self.db.set_validator_state("competition_version", str(current_version))
                return

            # Load last processed block
            block_str = await self.db.get_validator_state("last_processed_block")
            if block_str:
                self.last_processed_block = int(block_str)
                logger.info(f"Loaded state: last_processed_block={self.last_processed_block}")

            # Load evaluated code URLs
            urls_json = await self.db.get_validator_state("evaluated_code_urls")
            if urls_json:
                loaded = json.loads(urls_json)
                # Handle both old format (list of URLs) and new format (dict)
                if isinstance(loaded, list):
                    # Old format: convert to dict with placeholder values
                    self.evaluated_code_urls = {url: (0, "unknown") for url in loaded}
                elif isinstance(loaded, dict):
                    # New format: dict mapping URL -> [reveal_block, hotkey]
                    self.evaluated_code_urls = {url: tuple(info) for url, info in loaded.items()}
                else:
                    self.evaluated_code_urls = {}
                logger.info(f"Loaded state: {len(self.evaluated_code_urls)} evaluated URLs")
        except Exception as e:
            logger.warning(f"Failed to load state (starting fresh): {e}")
            self.last_processed_block = 0
            self.evaluated_code_urls = {}

    async def _save_state(self) -> None:
        """Persist validator state to database."""
        from crusades import COMPETITION_VERSION

        try:
            await self.db.set_validator_state(
                "competition_version", str(COMPETITION_VERSION)
            )
            await self.db.set_validator_state(
                "last_processed_block", str(self.last_processed_block)
            )
            await self.db.set_validator_state(
                "evaluated_code_urls",
                json.dumps({url: list(info) for url, info in self.evaluated_code_urls.items()}),
            )
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self._save_state()
        await super().cleanup()
        if self.db:
            await self.db.close()


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = argparse.ArgumentParser(description="Crusades Validator (URL-Based)")
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
        "--skip-blockchain-check",
        action="store_true",
        help="Skip blockchain registration check",
    )
    parser.add_argument(
        "--affinetes-mode",
        type=str,
        choices=["docker", "basilica"],
        default="docker",
        help="Execution mode: docker (local) or basilica (remote GPU)",
    )

    args = parser.parse_args()

    wallet = bt.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)

    validator = Validator(
        wallet=wallet,
        skip_blockchain_check=args.skip_blockchain_check,
        affinetes_mode=args.affinetes_mode,
    )

    logger.info("Starting validator")
    logger.info(f"   Affinetes mode: {args.affinetes_mode}")

    asyncio.run(validator.start())


if __name__ == "__main__":
    main()
