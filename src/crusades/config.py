"""Configuration management for templar-crusades (Chi/Affinetes Architecture)."""

import json
from functools import cache
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@cache
def get_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")


class StorageConfig(BaseModel):
    """Storage settings."""

    database_url: str = "sqlite+aiosqlite:///crusades.db"


class VerificationConfig(BaseModel):
    """Verification settings using gradient-based checks."""

    max_loss_difference: float = 0.5
    min_params_changed_ratio: float = 0.8
    gradient_cosine_min: float = 0.8
    gradient_norm_ratio_min: float = 0.5
    gradient_norm_ratio_max: float = 2.0


class MFUConfig(BaseModel):
    """MFU (Model FLOPs Utilization) calculation settings."""

    gpu_peak_tflops: float = 312.0  # A100 80GB peak TFLOPS (bfloat16)


class AdaptiveThresholdConfig(BaseModel):
    """Adaptive decay threshold for leaderboard replacement.

    Instead of a fixed 1% threshold, we use an adaptive threshold that:
    - Sets threshold = improvement when new leader wins
    - Decays over time towards base_threshold (loses decay_percent each interval)
    """

    base_threshold: float = 0.01  # Minimum threshold (1%)
    decay_percent: float = 0.05  # Percent to lose per interval (5% = loses 5% of excess)
    decay_interval_blocks: int = 100  # Blocks between decay steps (~20 min)


class DockerConfig(BaseModel):
    """Docker execution settings for validator evaluations.

    GPU device options:
    - "all": Use all available GPUs (default)
    - "0": Use only GPU 0
    - "0,1": Use GPUs 0 and 1
    - "none": Disable GPU (CPU only)
    """

    gpu_devices: str = "all"  # "all", "0", "0,1", "none"
    memory_limit: str = "32g"  # Docker memory limit
    shm_size: str = "8g"  # Shared memory size (important for PyTorch)


class BasilicaConfig(BaseModel):
    """Basilica cloud GPU settings for remote evaluation.

    Basilica is used for production evaluation when local GPUs
    are not available or for distributed validator setups.

    Prerequisites:
    1. Build and push image: docker push ghcr.io/org/templar-eval:latest
    2. Set BASILICA_API_TOKEN environment variable

    GPU Configuration:
    - gpu_count: Number of GPUs (1-8)
    - gpu_models: Acceptable GPU types (e.g., ["A100", "H100"])
    - min_gpu_memory_gb: Minimum GPU VRAM in GB
    """

    image: str = "ghcr.io/one-covenant/templar-eval:latest"  # Docker image in registry
    ttl_seconds: int = 3600  # Deployment TTL (1 hour default)
    gpu_count: int = 1  # Number of GPUs (1-8)
    gpu_models: list[str] = ["A100"]  # GPU types
    min_gpu_memory_gb: int = 40  # Minimum GPU memory in GB
    cpu: str = "4"  # CPU cores
    memory: str = "32Gi"  # Memory limit


class HParams(BaseModel):
    """Hyperparameters loaded from hparams.json.

    URL-Based Architecture:
    - Miners host train.py at any URL and commit the URL to blockchain
    - Validators download from miner's URL and evaluate via Docker/Basilica
    - All settings are defined in hparams.json (no hardcoded defaults)

    Versioning:
    - Competition version is derived from __version__ major number
    - Use crusades.COMPETITION_VERSION to get the current competition version
    """

    # Network settings
    netuid: int

    # Emissions distribution
    burn_rate: float
    burn_uid: int

    # Evaluation settings
    evaluation_runs: int
    eval_steps: int
    eval_timeout: int
    min_success_rate: float = 0.5  # Minimum success rate for submissions

    # Benchmark settings
    benchmark_model_name: str
    benchmark_dataset_name: str
    benchmark_dataset_split: str
    benchmark_data_samples: int
    benchmark_master_seed: int
    benchmark_sequence_length: int
    benchmark_batch_size: int

    # Timing settings
    set_weights_interval_blocks: int  # Minimum blocks between weight updates

    # Commitment settings (timelock encrypted via drand)
    reveal_blocks: int
    min_blocks_between_commits: int
    block_time: int

    # Docker execution settings (local GPU)
    docker: DockerConfig = Field(default_factory=DockerConfig)

    # Basilica settings (remote GPU)
    basilica: BasilicaConfig = Field(default_factory=BasilicaConfig)

    # Verification (nested config with defaults from JSON)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)

    # MFU calculation settings
    mfu: MFUConfig = Field(default_factory=MFUConfig)

    # Adaptive threshold for leaderboard
    adaptive_threshold: AdaptiveThresholdConfig = Field(default_factory=AdaptiveThresholdConfig)

    # Storage (for evaluation records - not in hparams.json)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Self:
        """Load hyperparameters from JSON file.

        Args:
            path: Path to hparams.json file

        Returns:
            HParams instance with all values from JSON

        Raises:
            FileNotFoundError: If hparams.json doesn't exist
            ValidationError: If required fields are missing
        """
        if path is None:
            # Default to hparams/hparams.json relative to project root
            path = get_project_root() / "hparams" / "hparams.json"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"hparams.json not found at {path}")

        with open(path) as f:
            data = json.load(f)

        return cls.model_validate(data)


class Config(BaseSettings):
    """Runtime configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CRUSADES_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Bittensor settings
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    subtensor_network: str = Field(
        default="finney",
        validation_alias="SUBTENSOR_NETWORK",  # Accept SUBTENSOR_NETWORK env var directly
    )

    # Paths
    hparams_path: str = "hparams/hparams.json"

    # Debug
    debug: bool = False


# Global instances (lazy loaded)
_hparams: HParams | None = None
_config: Config | None = None


def get_hparams() -> HParams:
    """Get or create global HParams instance."""
    global _hparams
    if _hparams is None:
        config = get_config()
        _hparams = HParams.load(config.hparams_path)
    return _hparams


def get_config() -> Config:
    """Get or create global Config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
