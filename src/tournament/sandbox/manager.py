"""Docker sandbox manager for executing miner training code.

SECURITY: Runs untrusted code in isolated containers with:
- No network access
- Read-only filesystem
- Resource limits (16GB RAM, 4 CPUs, 1 GPU)
- External time measurement (host-side timing)
- Random seed per evaluation (prevents pre-computation)

FAIRNESS: All miners evaluated with:
- Same 8B model (from hparams.json)
- Same dataset (from hparams.json)
- Same hardware limits
- External timing (can't be manipulated)
"""

import asyncio
import gc
import logging
import shutil
import tempfile
import time
from pathlib import Path

import docker
import torch
from docker.models.containers import Container

from ..config import get_hparams
from ..core.exceptions import SandboxError, SandboxTimeoutError
from ..core.protocols import SandboxResult
from ..schemas import BenchmarkConfig, SandboxOutput

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages Docker containers for running untrusted miner code.
    
    Mounts official 8B model and dataset (read-only) into sandbox.
    Model/data paths come from config (benchmark_model_path, benchmark_data_path).
    """

    IMAGE_NAME = "tournament-sandbox:latest"
    
    # GPU assignment: GPU 0 reserved for reference, GPU 1-7 for sandbox evaluations
    SANDBOX_GPUS = list(range(1, 8))  # GPUs 1, 2, 3, 4, 5, 6, 7
    _gpu_counter = 0  # Round-robin counter

    def __init__(
        self,
        benchmark_model_path: str,
        benchmark_data_path: str,
    ):
        """Initialize sandbox manager.
        
        Args:
            benchmark_model_path: Path to official 7B model directory
            benchmark_data_path: Path to official dataset file
            
        GPU Assignment:
            - GPU 0: Reserved for reference execution (always)
            - GPU 1-7: Round-robin for miner evaluations (parallel)
        """
        self.benchmark_model_path = Path(benchmark_model_path).resolve()
        self.benchmark_data_path = Path(benchmark_data_path).resolve()
        self.hparams = get_hparams()
        self._client: docker.DockerClient | None = None
        self._image_built = False

    @property
    def client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def initialize(self) -> None:
        """Initialize sandbox manager (build image if needed)."""
        await self._build_image()

    async def _build_image(self) -> None:
        """Build the sandbox Docker image."""
        if self._image_built:
            return

        # Check if image already exists
        try:
            self.client.images.get(self.IMAGE_NAME)
            logger.info(f"Docker image {self.IMAGE_NAME} already exists")
            self._image_built = True
            return
        except docker.errors.ImageNotFound:
            pass  # Image doesn't exist, need to build

        dockerfile_path = Path(__file__).parent / "Dockerfile"
        if not dockerfile_path.exists():
            raise SandboxError(f"Dockerfile not found at {dockerfile_path}")

        # Run in thread pool to avoid blocking
        logger.info(f"Building Docker image: {self.IMAGE_NAME}")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.images.build(
                path=str(dockerfile_path.parent),
                tag=self.IMAGE_NAME,
                rm=True,
            ),
        )
        self._image_built = True
        logger.info(f"Docker image built: {self.IMAGE_NAME}")

    def _get_next_gpu(self) -> int:
        """Get next available GPU in round-robin fashion.
        
        Returns:
            GPU ID (1-7)
        """
        gpu_id = self.SANDBOX_GPUS[self._gpu_counter % len(self.SANDBOX_GPUS)]
        self._gpu_counter += 1
        return gpu_id

    async def run(
        self,
        code_path: str,
        timeout_seconds: int | None = None,
        num_steps: int | None = None,
        random_seed: int = 42,
    ) -> SandboxResult:
        """
        Run training code in sandbox and measure TPS.
        
        Assigns each evaluation to a different GPU (1-7) in round-robin.
        GPU 0 is reserved for reference execution.

        Args:
            code_path: Path to the miner's train.py file
            timeout_seconds: Maximum execution time
            num_steps: Number of training steps to run
            random_seed: Random seed for reproducibility

        Returns:
            SandboxResult with TPS and other metrics
        """
        if timeout_seconds is None:
            timeout_seconds = self.hparams.eval_timeout
        if num_steps is None:
            num_steps = self.hparams.eval_steps

        code_path = Path(code_path).resolve()
        if not code_path.exists():
            raise SandboxError(f"Code file not found: {code_path}")

        # Assign GPU for this evaluation (round-robin across GPUs 1-7)
        assigned_gpu = self._get_next_gpu()
        logger.info(f"🎯 Assigning evaluation to GPU {assigned_gpu}")
        logger.info(f"📁 Code path: {code_path}")
        logger.info(f"⚙️  Config: {num_steps} steps, seed={random_seed}, timeout={timeout_seconds}s")

        # Create temporary directory for sandbox I/O
        temp_dir = Path(tempfile.mkdtemp(prefix="sandbox_"))

        try:
            # Copy code to temp directory
            shutil.copy(code_path, temp_dir / "train.py")

            # Write benchmark config
            config = BenchmarkConfig(
                model_path="/benchmark/model",
                data_path="/benchmark/data/test.pt",  # Full path to data file
                sequence_length=self.hparams.benchmark_sequence_length,
                batch_size=self.hparams.benchmark_batch_size,
                num_steps=num_steps,
                random_seed=random_seed,
            )
            (temp_dir / "config.json").write_text(config.model_dump_json())

            # Create output directory
            output_dir = temp_dir / "output"
            output_dir.mkdir()

            # Run container with external timing on assigned GPU
            result = await self._run_container(
                temp_dir=temp_dir,
                output_dir=output_dir,
                timeout_seconds=timeout_seconds,
                assigned_gpu=assigned_gpu,
            )

            return result

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def _run_container(
        self,
        temp_dir: Path,
        output_dir: Path,
        timeout_seconds: int,
        assigned_gpu: int,
    ) -> SandboxResult:
        """Run the Docker container on assigned GPU and collect results.
        
        Args:
            temp_dir: Temporary directory with code and config
            output_dir: Directory for results
            timeout_seconds: Maximum execution time
            assigned_gpu: GPU ID to use for this evaluation (1-7)
        """
        sandbox_config = self.hparams.sandbox

        container: Container | None = None
        start_time = time.perf_counter()

        try:
            # Create container with specific GPU assignment
            logger.info(f"🐳 Creating Docker container on GPU {assigned_gpu}...")
            logger.info(f"   Memory limit: {sandbox_config.memory_limit}, CPUs: {sandbox_config.cpu_count}")
            
            loop = asyncio.get_event_loop()
            # Use gpus parameter - simpler and more reliable
            container = await loop.run_in_executor(
                None,
                lambda: self.client.containers.run(
                    image=self.IMAGE_NAME,
                    detach=True,
                    network_mode="none",  # No network access
                    mem_limit=sandbox_config.memory_limit,
                    nano_cpus=int(sandbox_config.cpu_count * 1e9),
                    pids_limit=sandbox_config.pids_limit,
                    read_only=True,
                    tmpfs={
                        "/tmp": "size=1G,mode=1777",
                        "/root/.triton": "size=512M,mode=1777",  # For torch.compile cache
                        "/root/.cache": "size=512M,mode=1777",   # For other caches
                    },
                    volumes={
                        str(self.benchmark_model_path): {
                            "bind": "/benchmark/model",
                            "mode": "ro",
                        },
                        str(self.benchmark_data_path.parent): {
                            "bind": "/benchmark/data",
                            "mode": "ro",
                        },
                        str(temp_dir): {"bind": "/sandbox", "mode": "rw"},
                        str(output_dir): {"bind": "/output", "mode": "rw"},
                    },
                    user="0:0",  # Run as root to avoid permission issues
                    environment={
                        "CUDA_VISIBLE_DEVICES": "0",  # Inside container, mapped GPU appears as device 0
                    },
                    device_requests=[
                        docker.types.DeviceRequest(
                            device_ids=[str(assigned_gpu)],
                            capabilities=[["gpu"]],
                        )
                    ] if sandbox_config.gpu_count > 0 else [],
                ),
            )
            
            logger.info(f"✅ Container created: {container.id[:12]}")
            logger.info(f"⏳ Waiting for execution (timeout: {timeout_seconds}s)...")

            # Wait for container with timeout
            try:
                exit_result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: container.wait()),
                    timeout=timeout_seconds,
                )
                exit_code = exit_result["StatusCode"]
                logger.info(f"🏁 Container finished with exit code: {exit_code}")
            except TimeoutError:
                # Kill container on timeout
                logger.error(f"⏰ Container timed out after {timeout_seconds}s - killing...")
                await loop.run_in_executor(None, lambda: container.stop(timeout=5))
                wall_time = time.perf_counter() - start_time
                raise SandboxTimeoutError(f"Execution timed out after {wall_time:.1f}s")

            wall_time = time.perf_counter() - start_time
            logger.info(f"⏱️  Walltime: {wall_time:.2f}s")

            # Get container logs
            stdout = await loop.run_in_executor(
                None, lambda: container.logs(stdout=True, stderr=False).decode()
            )
            stderr = await loop.run_in_executor(
                None, lambda: container.logs(stdout=False, stderr=True).decode()
            )

            # Parse output
            logger.info(f"📊 Parsing results from {output_dir}/result.json...")
            output_file = output_dir / "result.json"
            if output_file.exists():
                output = SandboxOutput.model_validate_json(output_file.read_text())
                logger.info(f"✅ Results parsed: {output.total_tokens} tokens in {wall_time:.2f}s")
            else:
                # No output file - execution failed
                logger.error(f"❌ No result.json found - execution failed")
                logger.error(f"Exit code: {exit_code}")
                logger.error(f"Stdout: {stdout[:500] if stdout else 'None'}")
                logger.error(f"Stderr: {stderr[:500] if stderr else 'None'}")
                return SandboxResult(
                    success=False,
                    tokens_per_second=0.0,
                    total_tokens=0,
                    wall_time_seconds=wall_time,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    error=f"No output file. Exit code: {exit_code}",
                )

            if not output.success:
                return SandboxResult(
                    success=False,
                    tokens_per_second=0.0,
                    total_tokens=output.total_tokens,
                    wall_time_seconds=wall_time,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    error=output.error,
                )

            # Calculate TPS using external timing
            tps = output.total_tokens / wall_time if wall_time > 0 else 0.0

            # Load logits if available for verification
            final_logits = None
            final_logits_path = None
            if output.final_logits_path:
                logits_file = output_dir / "final_logits.pt"
                if logits_file.exists():
                    final_logits = torch.load(logits_file, weights_only=True)
                    final_logits_path = str(logits_file)

            result = SandboxResult(
                success=True,
                tokens_per_second=tps,
                total_tokens=output.total_tokens,
                wall_time_seconds=wall_time,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                final_loss=output.final_loss,
                final_logits=final_logits,
                final_logits_path=final_logits_path,
            )
            
            # Cleanup memory after sandbox
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return result

        except SandboxTimeoutError:
            raise
        except Exception as e:
            wall_time = time.perf_counter() - start_time
            return SandboxResult(
                success=False,
                tokens_per_second=0.0,
                total_tokens=0,
                wall_time_seconds=wall_time,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                error=str(e),
            )
        finally:
            # Cleanup container
            if container is not None:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: container.remove(force=True))
                except Exception:
                    pass

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
