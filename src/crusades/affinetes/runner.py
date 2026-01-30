"""Affinetes runner for evaluating miner submissions.

URL-Based Architecture:
- Miners host train.py at any URL (Gist, raw GitHub, etc.)
- Validator downloads code from committed URL
- Code is passed directly to the evaluation environment

Execution modes:
1. Docker mode - Local GPU evaluation via Docker container
2. Basilica mode - Remote cloud GPU evaluation via Basilica SDK

Environment Variables:
- BASILICA_API_TOKEN: API token for Basilica cloud GPU service
- VALIDATOR_EVAL_IMAGE: Docker image for local evaluation (default: templar-eval:latest)
- BASILICA_EVAL_IMAGE: Docker image for Basilica (default: ghcr.io/one-covenant/templar-eval:latest)
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx

# Optional: Basilica SDK for cloud GPU evaluation
try:
    from basilica import BasilicaClient

    BASILICA_AVAILABLE = True
except ImportError:
    BasilicaClient = None
    BASILICA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Basilica deployment cache (reuse deployments within TTL)
_basilica_deployment = None
_basilica_deployment_time = 0


@dataclass
class EvaluationResult:
    """Result from evaluating a miner's submission."""

    success: bool
    tps: float = 0.0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    error: str | None = None
    seed: str = ""
    task_id: int = 0
    diagnostics: dict = field(default_factory=dict)
    code: str | None = None  # Miner's code for storage

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        """Create from dictionary response."""
        return cls(
            success=data.get("success", False),
            tps=float(data.get("tps", 0.0)),
            total_tokens=int(data.get("total_tokens", 0)),
            wall_time_seconds=float(data.get("wall_time_seconds", 0.0)),
            error=data.get("error"),
            seed=str(data.get("seed", "")),
            task_id=int(data.get("task_id", 0)),
            diagnostics=data.get("diagnostics", {}),
            code=data.get("code"),
        )

    @classmethod
    def failure(cls, error: str, task_id: int = 0) -> "EvaluationResult":
        """Create a failure result."""
        return cls(success=False, error=error, task_id=task_id)


class AffinetesRunner:
    """Runs evaluations via Docker or Basilica.

    URL-Based Architecture:
    - Miner hosts train.py at any URL
    - Validator downloads code from committed URL
    - Code is passed directly to the evaluation container

    Example:
        runner = AffinetesRunner(mode="docker")
        result = await runner.evaluate(
            code="def inner_steps(...): ...",
            seed="12345",
        )
        if result.success:
            print(f"TPS: {result.tps}")
    """

    # Default Docker image for local evaluation
    DEFAULT_DOCKER_IMAGE = os.getenv("VALIDATOR_EVAL_IMAGE", "templar-eval:latest")

    # Default Basilica image (must be pushed to registry like ghcr.io)
    DEFAULT_BASILICA_IMAGE = os.getenv(
        "BASILICA_EVAL_IMAGE", "ghcr.io/one-covenant/templar-eval:latest"
    )

    def __init__(
        self,
        mode: Literal["docker", "basilica"] = "docker",
        basilica_endpoint: str | None = None,
        basilica_api_key: str | None = None,
        docker_gpu_devices: str = "all",
        docker_memory_limit: str = "32g",
        docker_shm_size: str = "8g",
        timeout: int = 600,
        model_url: str | None = None,
        data_url: str | None = None,
        output_tolerance: float = 0.02,
        loss_ratio_min: float = 0.8,
        loss_ratio_max: float = 1.2,
        validator_image: str | None = None,
        # Basilica-specific settings
        basilica_image: str | None = None,
        basilica_ttl_seconds: int = 3600,
        basilica_gpu_count: int = 1,
        basilica_gpu_models: list[str] | None = None,
        basilica_min_gpu_memory_gb: int = 40,
        basilica_cpu: str = "4",
        basilica_memory: str = "32Gi",
    ):
        """Initialize the runner.

        Args:
            mode: Execution mode ("docker" for local, "basilica" for remote)
            basilica_endpoint: Basilica API endpoint (not needed with SDK)
            basilica_api_key: Basilica API key (or BASILICA_API_TOKEN env var)
            docker_gpu_devices: GPU devices for Docker ("all", "0", "0,1", "none")
            docker_memory_limit: Docker memory limit (e.g., "32g")
            docker_shm_size: Shared memory size for Docker (e.g., "8g")
            timeout: Evaluation timeout in seconds
            model_url: Default model URL (HuggingFace model ID)
            data_url: Default data URL (HuggingFace dataset)
            output_tolerance: Verification tolerance (0.02 = 2%)
            loss_ratio_min: Minimum allowed loss ratio (default 0.8)
            loss_ratio_max: Maximum allowed loss ratio (default 1.2)
            validator_image: Docker image for local evaluation
            basilica_image: Docker image for Basilica (must be in registry)
            basilica_ttl_seconds: TTL for Basilica deployment (default 1 hour)
            basilica_gpu_count: Number of GPUs (1-8)
            basilica_gpu_models: Acceptable GPU models (e.g., ["A100", "H100"])
            basilica_min_gpu_memory_gb: Minimum GPU memory in GB
            basilica_cpu: CPU limit (e.g., "4")
            basilica_memory: Memory limit (e.g., "32Gi")
        """
        self.mode = mode
        self.basilica_endpoint = basilica_endpoint or os.getenv("BASILICA_ENDPOINT")
        self.basilica_api_key = basilica_api_key or os.getenv("BASILICA_API_TOKEN")
        self.docker_gpu_devices = docker_gpu_devices
        self.docker_memory_limit = docker_memory_limit
        self.docker_shm_size = docker_shm_size
        self.timeout = timeout
        self.default_model_url = model_url
        self.default_data_url = data_url
        self.output_tolerance = output_tolerance
        self.loss_ratio_min = loss_ratio_min
        self.loss_ratio_max = loss_ratio_max
        self.validator_image = validator_image or self.DEFAULT_DOCKER_IMAGE
        self.basilica_image = basilica_image or self.DEFAULT_BASILICA_IMAGE
        self.basilica_ttl_seconds = basilica_ttl_seconds
        self.basilica_gpu_count = basilica_gpu_count
        self.basilica_gpu_models = basilica_gpu_models or ["A100", "H100"]
        self.basilica_min_gpu_memory_gb = basilica_min_gpu_memory_gb
        self.basilica_cpu = basilica_cpu
        self.basilica_memory = basilica_memory

        if mode == "basilica":
            if not self.basilica_api_key:
                logger.warning("Basilica mode: BASILICA_API_TOKEN not set")
            logger.info("Basilica mode initialized")
            logger.info(f"   Image: {self.basilica_image}")
            logger.info(f"   TTL: {self.basilica_ttl_seconds}s")
            logger.info(f"   GPU: {self.basilica_gpu_count}x {self.basilica_gpu_models}")
            logger.info(f"   Min GPU Memory: {self.basilica_min_gpu_memory_gb}GB")
            logger.info(f"   CPU/Memory: {self.basilica_cpu} / {self.basilica_memory}")

    async def evaluate(
        self,
        code: str,
        seed: str | int = 0,
        model_url: str | None = None,
        data_url: str | None = None,
        steps: int = 5,
        batch_size: int = 8,
        sequence_length: int = 1024,
        data_samples: int = 10000,
        task_id: int = 0,
    ) -> EvaluationResult:
        """Evaluate a miner's train.py code.

        Args:
            code: Miner's train.py code (already downloaded from URL)
            seed: Random seed for evaluation
            model_url: HuggingFace model name
            data_url: HuggingFace dataset name
            steps: Number of training steps
            batch_size: Batch size
            sequence_length: Sequence length
            data_samples: Number of data samples
            task_id: Evaluation task identifier

        Returns:
            EvaluationResult with TPS score
        """
        model_url = model_url or self.default_model_url
        data_url = data_url or self.default_data_url

        if not model_url or not data_url:
            return EvaluationResult.failure(
                "model_url and data_url are required",
                task_id=task_id,
            )

        if not code or "def inner_steps" not in code:
            return EvaluationResult.failure(
                "Invalid code: must contain 'def inner_steps' function",
                task_id=task_id,
            )

        if self.mode == "docker":
            return await self._evaluate_docker(
                code=code,
                seed=str(seed),
                model_url=model_url,
                data_url=data_url,
                steps=steps,
                batch_size=batch_size,
                sequence_length=sequence_length,
                data_samples=data_samples,
                task_id=task_id,
            )
        elif self.mode == "basilica":
            return await self._evaluate_basilica(
                code=code,
                seed=str(seed),
                model_url=model_url,
                data_url=data_url,
                steps=steps,
                batch_size=batch_size,
                sequence_length=sequence_length,
                data_samples=data_samples,
                task_id=task_id,
            )
        else:
            return EvaluationResult.failure(
                f"Unknown mode: {self.mode}",
                task_id=task_id,
            )

    async def _evaluate_docker(
        self,
        code: str,
        seed: str,
        model_url: str,
        data_url: str,
        steps: int,
        batch_size: int,
        sequence_length: int,
        data_samples: int,
        task_id: int,
    ) -> EvaluationResult:
        """Run evaluation locally using Docker.

        Code is mounted directly into the container - no downloads needed.
        """
        logger.info("Running Docker evaluation")
        logger.info(f"   Code size: {len(code)} bytes")

        # Check if validator image exists
        check_cmd = ["docker", "image", "inspect", self.validator_image]
        check_result = subprocess.run(check_cmd, capture_output=True)

        if check_result.returncode != 0:
            return EvaluationResult.failure(
                f"Validator image not found: {self.validator_image}. "
                f"Build it first: cd environments/templar && docker build -t {self.validator_image} .",
                task_id=task_id,
            )

        # Write miner's code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            prefix="train_",
        ) as f:
            f.write(code)
            train_path = f.name

        # Make readable by container's non-root user
        os.chmod(train_path, 0o644)

        # Create evaluation script that reads code from mounted file
        eval_script = f'''
import asyncio
import json
import sys
sys.path.insert(0, '/app')

from env import Actor

async def main():
    # Read miner's code
    with open('/app/scripts/miner_train.py') as f:
        code = f.read()

    actor = Actor()
    result = await actor.evaluate(
        task_id={task_id},
        seed="{seed}",
        model_url="{model_url}",
        data_url="{data_url}",
        steps={steps},
        batch_size={batch_size},
        sequence_length={sequence_length},
        data_samples={data_samples},
        timeout={self.timeout},
        code=code,
        output_tolerance={self.output_tolerance},
        loss_ratio_min={self.loss_ratio_min},
        loss_ratio_max={self.loss_ratio_max},
    )
    print("EVAL_RESULT:" + json.dumps(result))

asyncio.run(main())
'''

        # Write eval script to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(eval_script)
            script_path = f.name

        # Make readable by container's non-root user
        os.chmod(script_path, 0o644)

        try:
            # Build Docker run command
            # NOTE: Mount to /app/scripts/ (not /tmp/) because we use --tmpfs on /tmp
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{script_path}:/app/scripts/eval_script.py:ro",
                "-v",
                f"{train_path}:/app/scripts/miner_train.py:ro",
            ]

            # Add GPU if configured
            if self.docker_gpu_devices and self.docker_gpu_devices.lower() != "none":
                if self.docker_gpu_devices.lower() == "all":
                    docker_cmd.extend(["--gpus", "all"])
                else:
                    docker_cmd.extend(["--gpus", f'"device={self.docker_gpu_devices}"'])

            # Memory limits (from hparams.json docker config)
            docker_cmd.extend(
                [
                    "--memory",
                    self.docker_memory_limit,
                    "--shm-size",
                    self.docker_shm_size,
                ]
            )

            # =================================================================
            # SECURITY SANDBOX - Protect against malicious miner code
            # =================================================================
            docker_cmd.extend(
                [
                    # NETWORK ISOLATION - Prevent miner code from making any network requests
                    # Model and data are pre-cached in the Docker image
                    "--network",
                    "none",
                    # Drop all Linux capabilities
                    "--cap-drop",
                    "ALL",
                    # Prevent privilege escalation
                    "--security-opt",
                    "no-new-privileges",
                    # Read-only root filesystem
                    "--read-only",
                    # Limit number of processes (prevent fork bombs)
                    # PyTorch with OpenMP needs many threads, 1024 is reasonable
                    "--pids-limit",
                    "1024",
                    # Writable /tmp for temporary files (exec needed for torch.compile)
                    "--tmpfs",
                    "/tmp:rw,exec,nosuid,size=4g",
                    # Writable Triton cache for torch.compile kernels (exec needed for .so files)
                    "--tmpfs",
                    "/home/appuser/.triton:rw,exec,size=2g",
                    # NOTE: Don't mount tmpfs on ~/.cache/huggingface - model is pre-cached there!
                ]
            )

            # Environment variables
            docker_cmd.extend(
                [
                    "-e",
                    f"OUTPUT_VECTOR_TOLERANCE={self.output_tolerance}",
                ]
            )

            # Timeout
            docker_cmd.extend(
                [
                    "--stop-timeout",
                    str(self.timeout),
                ]
            )

            # Image and command
            docker_cmd.extend(
                [
                    self.validator_image,
                    "python",
                    "/app/scripts/eval_script.py",
                ]
            )

            logger.info(f"Running evaluation in {self.validator_image}...")
            logger.debug(f"   Full Docker command: {' '.join(docker_cmd)}")
            logger.info(f"   Docker command: {' '.join(docker_cmd[:6])}...")

            # Run with timeout - stream logs in real-time
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout for unified logging
            )

            # Stream logs in real-time and collect for parsing
            # Use chunk-based reading to handle long lines (progress bars, etc.)
            stdout_lines = []

            def _should_log(line: str) -> bool:
                """Filter out noisy logs, only show important ones."""
                # Always skip these
                if not line or line.startswith("EVAL_RESULT:"):
                    return False
                # Skip HTTP request logs
                if "HTTP Request:" in line or "httpx" in line:
                    return False
                # Skip progress bars
                if "Loading weights:" in line or "Fetching" in line:
                    return False
                # Skip deprecation/warning noise
                if "is deprecated" in line or "UserWarning" in line:
                    return False
                # Skip HuggingFace download noise
                if "huggingface" in line.lower() and "INFO" in line:
                    return False
                # Always show important logs
                if any(
                    kw in line
                    for kw in [
                        "VERIFICATION",
                        "CHECK",
                        "PASSED",
                        "FAILED",
                        "ERROR",
                        "error",
                        "Exception",
                        "Traceback",
                        "env |",  # env.py logs
                    ]
                ):
                    return True
                # Show other logs at debug level only
                return False

            try:

                async def read_stream():
                    buffer = ""
                    while True:
                        # Read in chunks to avoid buffer limit issues with long lines
                        chunk = await asyncio.wait_for(
                            process.stdout.read(8192),  # 8KB chunks
                            timeout=self.timeout + 60,
                        )
                        if not chunk:
                            # Process remaining buffer at end
                            if buffer:
                                for line in buffer.split("\n"):
                                    line = line.rstrip()
                                    if line:
                                        stdout_lines.append(line)
                                        if _should_log(line):
                                            logger.info(f"   [DOCKER] {line}")
                            break

                        buffer += chunk.decode()
                        # Process complete lines as they arrive
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.rstrip()
                            stdout_lines.append(line)
                            if _should_log(line):
                                logger.info(f"   [DOCKER] {line}")

                await read_stream()
                await process.wait()

            except TimeoutError:
                process.kill()
                return EvaluationResult.failure(
                    f"Evaluation timed out after {self.timeout}s",
                    task_id=task_id,
                )

            stdout_text = "\n".join(stdout_lines)

            if process.returncode != 0:
                # Log the output for debugging
                logger.error(f"Docker container failed with exit code {process.returncode}")
                if stdout_text:
                    logger.error(f"Docker output: {stdout_text[:500]}")
                return EvaluationResult.failure(
                    f"Container failed with exit code: {process.returncode}. Output: {stdout_text[:200]}",
                    task_id=task_id,
                )

            # Parse result
            for line in stdout_text.split("\n"):
                if line.startswith("EVAL_RESULT:"):
                    result_json = line[len("EVAL_RESULT:") :]
                    try:
                        result_data = json.loads(result_json)
                        result = EvaluationResult.from_dict(result_data)
                        result.code = code  # Include code in result
                        return result
                    except json.JSONDecodeError as e:
                        return EvaluationResult.failure(
                            f"Invalid result JSON: {e}",
                            task_id=task_id,
                        )

            return EvaluationResult.failure(
                f"No evaluation result in output. stdout: {stdout_text[:200]}",
                task_id=task_id,
            )

        finally:
            try:
                os.unlink(script_path)
                os.unlink(train_path)
            except Exception:
                pass

    async def _evaluate_basilica(
        self,
        code: str,
        seed: str,
        model_url: str,
        data_url: str,
        steps: int,
        batch_size: int,
        sequence_length: int,
        data_samples: int,
        task_id: int,
    ) -> EvaluationResult:
        """Run evaluation remotely via Basilica SDK.

        Uses BasilicaClient to deploy a custom Docker image and call
        the /evaluate endpoint for TPS evaluation.

        Flow:
        1. Deploy image to Basilica (or reuse existing deployment)
        2. Wait for deployment to be ready (/health endpoint)
        3. POST to /evaluate with miner's code
        4. Return TPS results
        """
        logger.info("=" * 60)
        logger.info("[BASILICA] Starting remote GPU evaluation")
        logger.info("=" * 60)
        logger.info("[BASILICA] Configuration:")
        logger.info(f"   Image: {self.basilica_image}")
        logger.info(f"   Model: {model_url}")
        logger.info(f"   Dataset: {data_url}")
        logger.info(f"   Steps: {steps}, Batch size: {batch_size}")
        logger.info(f"   Task ID: {task_id}, Seed: {seed}")
        logger.info(f"   Code size: {len(code)} bytes")

        if not BASILICA_AVAILABLE:
            logger.error("[BASILICA] SDK not installed!")
            return EvaluationResult.failure(
                "basilica SDK not installed. Run: uv add basilica",
                task_id=task_id,
            )

        try:
            # Get or create Basilica deployment
            logger.info("[BASILICA] Acquiring deployment...")
            deployment = await self._get_basilica_deployment()

            if deployment is None:
                logger.error("[BASILICA] Failed to create deployment!")
                return EvaluationResult.failure(
                    "Failed to create Basilica deployment",
                    task_id=task_id,
                )

            logger.info("-" * 60)
            logger.info("[BASILICA] Deployment ready!")
            logger.info(f"   URL: {deployment.url}")
            if hasattr(deployment, "id"):
                logger.info(f"   Deployment ID: {deployment.id}")
            logger.info("-" * 60)

            # Check health endpoint first
            logger.info("[BASILICA] Checking health endpoint...")
            async with httpx.AsyncClient(timeout=30) as client:
                try:
                    health_response = await client.get(f"{deployment.url}/health")
                    if health_response.status_code == 200:
                        logger.info("[BASILICA] Health check: ✅ OK")
                    else:
                        logger.warning(f"[BASILICA] Health check: {health_response.status_code}")
                except Exception as e:
                    logger.warning(f"[BASILICA] Health check failed: {e}")

            # Call the /evaluate endpoint
            payload = {
                "task_id": task_id,
                "seed": seed,
                "model_url": model_url,
                "data_url": data_url,
                "steps": steps,
                "batch_size": batch_size,
                "timeout": self.timeout,
                "sequence_length": sequence_length,
                "data_samples": data_samples,
                "code": code,
                "output_tolerance": self.output_tolerance,
                "loss_ratio_min": self.loss_ratio_min,
                "loss_ratio_max": self.loss_ratio_max,
            }

            logger.info("[BASILICA] Sending evaluation request...")
            logger.info(f"   POST {deployment.url}/evaluate")
            logger.info(f"   Timeout: {self.timeout + 120}s")

            start_time = time.time()

            async with httpx.AsyncClient(timeout=self.timeout + 120) as client:
                response = await client.post(
                    f"{deployment.url}/evaluate",
                    json=payload,
                )

                elapsed = time.time() - start_time
                logger.info(f"[BASILICA] Response received in {elapsed:.1f}s")
                logger.info(f"   Status code: {response.status_code}")

                if response.status_code != 200:
                    error_text = response.text[:500]
                    logger.error("[BASILICA] Evaluation failed!")
                    logger.error(f"   Error: {error_text}")
                    return EvaluationResult.failure(
                        f"Basilica /evaluate error: {response.status_code} - {error_text}",
                        task_id=task_id,
                    )

                result_data = response.json()
                result = EvaluationResult.from_dict(result_data)
                result.code = code

                logger.info("=" * 60)
                logger.info("[BASILICA] Evaluation complete!")
                logger.info(f"   Success: {result.success}")
                logger.info(f"   TPS: {result.tps:,.2f} tokens/second")
                logger.info(f"   Total tokens: {result.total_tokens:,}")
                logger.info(f"   Wall time: {result.wall_time_seconds:.2f}s")
                if result.diagnostics:
                    logger.info(f"   Diagnostics: {result.diagnostics}")
                if result.error:
                    logger.error(f"   Error: {result.error}")
                logger.info("=" * 60)

                return result

        except TimeoutError:
            logger.error(f"[BASILICA] Timeout after {self.timeout}s!")
            return EvaluationResult.failure(
                f"Basilica timeout after {self.timeout}s",
                task_id=task_id,
            )
        except Exception as e:
            logger.error(f"[BASILICA] Error: {e}")
            logger.error(traceback.format_exc())
            return EvaluationResult.failure(
                f"Basilica error: {e}",
                task_id=task_id,
            )

    async def _get_basilica_deployment(self):
        """Get or create a Basilica deployment.

        Reuses existing deployment if within TTL, otherwise creates new one.
        """
        global _basilica_deployment, _basilica_deployment_time

        # Check if existing deployment is still valid
        now = time.time()
        ttl_buffer = 300  # 5 minute buffer before TTL expires

        if (
            _basilica_deployment is not None
            and now - _basilica_deployment_time < self.basilica_ttl_seconds - ttl_buffer
        ):
            remaining = self.basilica_ttl_seconds - (now - _basilica_deployment_time)
            logger.info("[BASILICA] Reusing existing deployment")
            logger.info(f"   URL: {_basilica_deployment.url}")
            logger.info(f"   TTL remaining: {remaining:.0f}s ({remaining / 60:.1f} min)")
            return _basilica_deployment

        # Create new deployment
        logger.info("[BASILICA] Creating NEW deployment (no valid cached deployment)")
        logger.info(f"   Image: {self.basilica_image}")
        logger.info(f"   GPU: {self.basilica_gpu_count}x {self.basilica_gpu_models}")
        logger.info(f"   Min GPU memory: {self.basilica_min_gpu_memory_gb}GB")
        logger.info(f"   CPU: {self.basilica_cpu}, Memory: {self.basilica_memory}")
        logger.info(
            f"   TTL: {self.basilica_ttl_seconds}s ({self.basilica_ttl_seconds / 60:.0f} min)"
        )
        logger.info("[BASILICA] Requesting GPU from Basilica... (this may take 2-5 minutes)")

        try:
            deploy_start = time.time()
            client = BasilicaClient()

            deployment = client.deploy(
                name="templar-eval",
                image=self.basilica_image,
                port=8000,
                ttl_seconds=self.basilica_ttl_seconds,
                timeout=300,  # Wait up to 5 min for deployment (GPU provisioning can be slow)
                # GPU configuration
                gpu_count=self.basilica_gpu_count,
                gpu_models=self.basilica_gpu_models,
                min_gpu_memory_gb=self.basilica_min_gpu_memory_gb,
                # Resource limits
                cpu=self.basilica_cpu,
                memory=self.basilica_memory,
            )

            deploy_time = time.time() - deploy_start
            logger.info(f"[BASILICA] ✅ Deployment created in {deploy_time:.1f}s!")
            logger.info(f"   Deployment URL: {deployment.url}")
            if hasattr(deployment, "id"):
                logger.info(f"   Deployment ID: {deployment.id}")
            logger.info(f"   GPU: {self.basilica_gpu_count}x {self.basilica_gpu_models}")
            logger.info(
                f"   TTL: {self.basilica_ttl_seconds}s (expires in {self.basilica_ttl_seconds / 60:.0f} min)"
            )

            _basilica_deployment = deployment
            _basilica_deployment_time = now

            return deployment

        except Exception as e:
            logger.error("[BASILICA] ❌ Failed to deploy!")
            logger.error(f"   Error: {e}")
            logger.error(traceback.format_exc())
            return None

    async def build_validator_image(self, env_path: Path | None = None) -> bool:
        """Build the validator's evaluation Docker image.

        Args:
            env_path: Path to environments/templar directory

        Returns:
            True if build succeeded
        """
        if env_path is None:
            candidates = [
                Path(__file__).parent.parent.parent.parent / "environments" / "templar",
                Path.cwd() / "environments" / "templar",
            ]
            for candidate in candidates:
                if candidate.exists() and (candidate / "Dockerfile").exists():
                    env_path = candidate
                    break

        if env_path is None or not env_path.exists():
            logger.error("Could not find environments/templar directory")
            return False

        logger.info(f"Building validator image: {self.validator_image}")
        logger.info(f"   From: {env_path}")

        cmd = [
            "docker",
            "build",
            "-t",
            self.validator_image,
            str(env_path),
        ]

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            logger.error("Failed to build validator image")
            return False

        logger.info(f"Successfully built: {self.validator_image}")
        return True


def create_runner(
    mode: str = "docker",
    **kwargs,
) -> AffinetesRunner:
    """Factory function to create an AffinetesRunner.

    Args:
        mode: "docker" or "basilica"
        **kwargs: Additional arguments

    Returns:
        Configured AffinetesRunner
    """
    if mode == "basilica":
        kwargs.setdefault("basilica_endpoint", os.getenv("BASILICA_ENDPOINT"))
        kwargs.setdefault("basilica_api_key", os.getenv("BASILICA_API_TOKEN"))

    return AffinetesRunner(mode=mode, **kwargs)
