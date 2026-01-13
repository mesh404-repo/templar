"""Configuration management for templar-tournament."""

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SandboxConfig(BaseModel):
    """Sandbox execution settings."""

    memory_limit: str = "16g"
    cpu_count: int = 4
    gpu_count: int = 1
    pids_limit: int = 256


class StorageConfig(BaseModel):
    """Storage settings."""

    database_url: str = "sqlite+aiosqlite:///tournament.db"


class AntiCopyingConfig(BaseModel):
    """Anti-copying protection settings."""

    submission_cooldown_minutes: int = 60
    hide_pending_submissions: bool = True


class VerificationConfig(BaseModel):
    """Verification tolerance settings."""

    output_vector_tolerance: float = 0.02  # 2% aggregate difference allowed
    deterministic_mode: bool = True


class HParams(BaseModel):
    """Hyperparameters loaded from hparams.json."""

    netuid: int = 3

    # Evaluation settings
    num_evals_per_submission: int = 1  # Number of validators that must evaluate
    evaluation_runs: int = 5  # Number of runs per submission (median taken for fairness)
    eval_steps: int = 100
    eval_timeout: int = 600

    # Benchmark settings - EXACT model and data everyone uses
    benchmark_model_name: str = "meta-llama/Llama-3.2-8B"
    benchmark_model_revision: str = "main"
    benchmark_dataset_name: str = "HuggingFaceFW/fineweb"
    benchmark_dataset_split: str = "train"
    benchmark_dataset_subset: str | None = None
    benchmark_sequence_length: int = 1024
    benchmark_batch_size: int = 8

    # Timing settings
    set_weights_interval_seconds: int = 600

    # Payment settings (anti-spam)
    submission_cost_rao: int = 100_000_000  # 0.1 TAO default

    # Nested configs
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    anti_copying: AntiCopyingConfig = Field(default_factory=AntiCopyingConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> Self:
        """Load hyperparameters from JSON file."""
        if path is None:
            # Default to hparams/hparams.json relative to project root
            path = Path(__file__).parent.parent.parent.parent / "hparams" / "hparams.json"
        else:
            path = Path(path)

        if not path.exists():
            return cls()

        with open(path) as f:
            data = json.load(f)

        return cls.model_validate(data)


class Config(BaseSettings):
    """Runtime configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="TOURNAMENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Bittensor settings
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    subtensor_network: str = "finney"

    # R2/S3 storage (for code submissions)
    r2_account_id: str = ""
    r2_bucket_name: str = "tournament-submissions"
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Paths (where official 7B model and data are stored after setup)
    hparams_path: str = "hparams/hparams.json"
    benchmark_model_path: str = "benchmark/model"  # HuggingFace model directory
    benchmark_data_path: str = "benchmark/data/test.pt"  # Test data (validators use for evaluation)
    # Note: Miners have train.pt, validators have test.pt (different seeds)

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
