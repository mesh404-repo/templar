"""Main verification orchestrator for comparing miner submissions against reference."""

import ast
import logging
import random
import tempfile
from pathlib import Path

import torch

from ..sandbox.manager import SandboxManager
from ..schemas import BenchmarkConfig, VerificationResult
from .config import VerificationConfig
from .errors import (
    LogitsMismatchError,
    LossMismatchError,
    SandboxExecutionError,
    TokenCountMismatchError,
    VerificationError,
)
from .reference import ReferenceExecutor, ReferenceResult

logger = logging.getLogger(__name__)


class SandboxVerifier:
    """Verifies miner submissions against reference execution.

    This class orchestrates the entire verification process:
    1. Run reference inner_steps to get expected outputs
    2. Run miner code in sandbox with same inputs
    3. Compare outputs and calculate TPS if valid

    Example:
        verifier = SandboxVerifier(sandbox_manager, reference_executor, config)
        result = await verifier.verify_and_benchmark(code_path)
        if result.success:
            print(f"TPS: {result.tokens_per_second}")
        else:
            print(f"Failed: {result.error_message}")
    """

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        reference_executor: ReferenceExecutor,
        config: VerificationConfig,
    ):
        """Initialize the verifier.

        Args:
            sandbox_manager: Manager for running code in Docker sandbox.
            reference_executor: Executor for running reference implementation.
            config: Verification tolerances and settings.
        """
        self.sandbox = sandbox_manager
        self.reference = reference_executor
        self.config = config

    async def verify_and_benchmark(
        self,
        code_path: str,
        seed: int | None = None,
        timeout_seconds: int | None = None,
        num_steps: int | None = None,
    ) -> VerificationResult:
        """Run verification and return TPS if valid.

        Args:
            code_path: Path to the miner's train.py file.
            seed: Random seed for this evaluation (generated randomly if None).
                  Same seed is used for both reference and miner execution to ensure
                  outputs match. Different seed per evaluation prevents pre-computation.
            timeout_seconds: Sandbox timeout (defaults to sandbox config).
            num_steps: Number of training steps (defaults to benchmark config).

        Returns:
            VerificationResult with success status and TPS if valid.
        """
        # Generate random seed if not provided (security: prevents pre-computation)
        # The SAME seed is used for both reference and miner in this evaluation
        seed = seed if seed is not None else random.randint(1, 2**31 - 1)

        logger.info(f"Using random seed: {seed} for this evaluation")

        try:
            # Step 1: Run reference execution
            logger.info(f"Step 1/4: Running reference execution (seed={seed})...")
            ref_result = self.reference.execute(seed=seed)
            logger.info(
                f"  Reference complete: {ref_result.total_tokens:,} tokens, "
                f"loss={ref_result.final_loss:.4f}"
            )

            # Step 2: Save reference outputs for sandbox comparison
            logger.info("Step 2/4: Preparing sandbox inputs...")
            with tempfile.TemporaryDirectory(prefix="verification_") as temp_dir:
                temp_path = Path(temp_dir)

                # Save reference data for sandbox
                self.reference.save_reference_outputs(ref_result, temp_path)

                # Step 3: Run miner code in sandbox
                logger.info("Step 3/4: Running miner code in sandbox...")
                sandbox_result = await self.sandbox.run(
                    code_path=code_path,
                    timeout_seconds=timeout_seconds,
                    num_steps=num_steps,
                    random_seed=seed,
                )

                # Check sandbox execution success
                if not sandbox_result.success:
                    logger.error(f"Sandbox execution failed: {sandbox_result.error}")
                    raise SandboxExecutionError(
                        exit_code=sandbox_result.exit_code,
                        stdout=sandbox_result.stdout,
                        stderr=sandbox_result.stderr,
                        error=sandbox_result.error,
                    )

                logger.info(
                    f"  Sandbox complete: {sandbox_result.total_tokens:,} tokens in "
                    f"{sandbox_result.wall_time_seconds:.2f}s"
                )

                # Step 4: Verify outputs
                logger.info("Step 4/4: Verifying outputs...")
                self._verify_outputs(ref_result, sandbox_result)

                # Calculate TPS
                tps = sandbox_result.total_tokens / sandbox_result.wall_time_seconds
                logger.info(f"VERIFICATION PASSED - TPS: {tps:,.2f}")

                return VerificationResult(
                    success=True,
                    tokens_per_second=tps,
                    total_tokens=sandbox_result.total_tokens,
                    wall_time_seconds=sandbox_result.wall_time_seconds,
                    final_loss=ref_result.final_loss,  # Use reference loss as baseline
                )

        except VerificationError as e:
            logger.error(f"Verification failed: {type(e).__name__}")
            logger.error(e.message)
            return VerificationResult(
                success=False,
                error_message=e.message,
                error_type=type(e).__name__,
            )

        except Exception as e:
            logger.exception(f"Unexpected error during verification: {e}")
            return VerificationResult(
                success=False,
                error_message=f"Unexpected error: {type(e).__name__}: {str(e)}",
                error_type="UnexpectedError",
            )

    def _verify_outputs(
        self,
        ref_result: ReferenceResult,
        sandbox_result,
    ) -> None:
        """Verify sandbox outputs match reference.

        Args:
            ref_result: Reference execution result.
            sandbox_result: Sandbox execution result.

        Raises:
            LogitsMismatchError: If output vectors don't match within tolerance.
            TokenCountMismatchError: If token counts don't match.
        """
        # 1. Check token count (exact match required)
        if ref_result.total_tokens != sandbox_result.total_tokens:
            raise TokenCountMismatchError(
                expected=ref_result.total_tokens,
                actual=sandbox_result.total_tokens,
            )
        logger.info(f"  [OK] Token count matches: {sandbox_result.total_tokens:,}")

        # 2. Check output vectors (aggregate difference within tolerance)
        if hasattr(sandbox_result, "final_logits") and sandbox_result.final_logits is not None:
            sandbox_logits = sandbox_result.final_logits
            if isinstance(sandbox_logits, str):
                # Load from path
                sandbox_logits = torch.load(sandbox_logits, weights_only=True)

            # Ensure same device for comparison
            ref_logits = ref_result.final_logits.to(sandbox_logits.device)

            # Calculate aggregate difference across all dimensions
            diff = (ref_logits - sandbox_logits).abs()
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()

            # Calculate as percentage of mean absolute value
            mean_abs_value = ref_logits.abs().mean().item()
            if mean_abs_value > 0:
                aggregate_diff_pct = mean_diff / mean_abs_value
            else:
                aggregate_diff_pct = mean_diff

            if aggregate_diff_pct > self.config.output_vector_tolerance:
                raise LogitsMismatchError(
                    expected=ref_logits,
                    actual=sandbox_logits,
                    tolerance=self.config.output_vector_tolerance,
                )

            logger.info(
                f"  [OK] Output vectors match within {self.config.output_vector_tolerance*100:.1f}%"
            )
            logger.info(f"      Mean diff: {mean_diff:.6f}, Max diff: {max_diff:.6f}")
            logger.info(f"      Aggregate: {aggregate_diff_pct*100:.2f}% (threshold: {self.config.output_vector_tolerance*100:.1f}%)")

    async def verify_code_structure(self, code_path: str) -> tuple[bool, str | None]:
        """Quick validation of code structure before full verification.

        Checks that the code:
        - Has valid Python syntax
        - Defines an inner_steps function
        - Has the correct signature

        Args:
            code_path: Path to the miner's train.py file.

        Returns:
            Tuple of (is_valid, error_message).
        """
        code_path = Path(code_path)
        if not code_path.exists():
            return False, f"Code file not found: {code_path}"

        code = code_path.read_text()

        # Check syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"

        # Find inner_steps function
        inner_steps_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "inner_steps":
                inner_steps_found = True
                # Check argument count (should have 5: model, data_iterator, optimizer, num_steps, device)
                args = node.args
                required_args = 5
                actual_args = len(args.args)
                if actual_args < required_args:
                    return False, (
                        f"inner_steps function has {actual_args} arguments, "
                        f"expected at least {required_args}"
                    )
                break

        if not inner_steps_found:
            return False, "Missing required function: inner_steps"

        return True, None


async def create_verifier(
    model_path: str,
    data_path: str,
    benchmark_config: BenchmarkConfig,
    verification_config: VerificationConfig | None = None,
) -> SandboxVerifier:
    """Factory function to create a configured verifier.

    Args:
        model_path: Path to benchmark model.
        data_path: Path to benchmark data.
        benchmark_config: Benchmark configuration.
        verification_config: Verification tolerances (uses defaults if None).

    Returns:
        Configured SandboxVerifier instance.
    """
    if verification_config is None:
        verification_config = VerificationConfig()

    # Create reference executor
    reference_executor = ReferenceExecutor(
        model_path=model_path,
        data_path=data_path,
        config=benchmark_config,
    )

    # Create sandbox manager
    sandbox_manager = SandboxManager(
        benchmark_model_path=model_path,
        benchmark_data_path=data_path,
    )
    await sandbox_manager.initialize()

    return SandboxVerifier(
        sandbox_manager=sandbox_manager,
        reference_executor=reference_executor,
        config=verification_config,
    )
