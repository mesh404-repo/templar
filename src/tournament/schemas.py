"""Pydantic schemas for API and data transfer."""

from datetime import datetime

from pydantic import BaseModel, Field

from .core.protocols import SubmissionStatus


class SubmissionCreate(BaseModel):
    """Schema for creating a new submission."""

    miner_hotkey: str
    miner_uid: int
    code_hash: str
    bucket_path: str


class SubmissionResponse(BaseModel):
    """Schema for submission API responses."""

    model_config = {"from_attributes": True}

    submission_id: str
    miner_hotkey: str
    miner_uid: int
    code_hash: str
    status: SubmissionStatus
    created_at: datetime
    updated_at: datetime
    final_score: float | None = None
    error_message: str | None = None


class EvaluationCreate(BaseModel):
    """Schema for creating a new evaluation result."""

    submission_id: str
    evaluator_hotkey: str
    tokens_per_second: float
    total_tokens: int
    wall_time_seconds: float
    success: bool
    error: str | None = None


class EvaluationResponse(BaseModel):
    """Schema for evaluation API responses."""

    model_config = {"from_attributes": True}

    evaluation_id: str
    submission_id: str
    evaluator_hotkey: str
    tokens_per_second: float
    total_tokens: int
    wall_time_seconds: float
    success: bool
    error: str | None = None
    created_at: datetime


class LeaderboardEntry(BaseModel):
    """Schema for leaderboard entries."""

    rank: int
    submission_id: str
    miner_hotkey: str
    miner_uid: int
    final_score: float
    num_evaluations: int
    created_at: datetime


class BenchmarkConfig(BaseModel):
    """Configuration passed to sandbox for benchmarking."""

    model_path: str
    data_path: str
    sequence_length: int = 1024
    batch_size: int = 8
    num_steps: int = 100
    random_seed: int = Field(default_factory=lambda: 42)


class SandboxOutput(BaseModel):
    """Output written by sandbox runner."""

    total_tokens: int
    num_steps: int
    success: bool
    error: str | None = None
    # New fields for verification
    final_logits_path: str | None = None  # Path to saved logits tensor
    final_loss: float | None = None


class InnerStepsResult(BaseModel):
    """Result returned by miner's inner_steps function.

    Miners must return this from their inner_steps function to pass verification.

    Attributes:
        final_logits: Output logits from the last forward pass.
                      Shape: (batch_size, seq_len, vocab_size)
        total_tokens: Total number of tokens processed across all steps.
        final_loss: Loss value from the last training step.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Note: final_logits is stored as a path in the sandbox output
    # because we can't serialize torch.Tensor directly in Pydantic
    total_tokens: int
    final_loss: float


class ReferenceResult(BaseModel):
    """Result from reference execution for verification comparison.

    Contains all the expected values that miner outputs must match.
    """

    total_tokens: int
    final_loss: float
    # Paths to serialized tensors
    final_logits_path: str | None = None
    initial_state_path: str | None = None


class VerificationResult(BaseModel):
    """Result of verification check."""

    success: bool
    tokens_per_second: float = 0.0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    final_loss: float = 0.0
    error_message: str | None = None
    error_type: str | None = None  # e.g., "LogitsMismatch", "TokenCountMismatch"
