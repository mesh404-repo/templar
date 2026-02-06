"""SQLAlchemy models for database persistence."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ..core.protocols import SubmissionStatus


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class SubmissionModel(Base):
    """Database model for code submissions.

    URL-Based Architecture:
    - submission_id: Format "commit_{block}_{uid}" for submissions
    - code_hash: Code URL (used as unique identifier)
    - bucket_path: Code URL (source location)
    - code_content: Actual miner code (stored after evaluation for conflict resolution)
    - spec_version: Version of validator that evaluated this submission

    All fields are preserved for future conflict resolution:
    - miner identity (hotkey, uid)
    - source location (code URL)
    - actual code content
    - timestamps
    - evaluation results
    """

    __tablename__ = "submissions"

    submission_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    miner_hotkey: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    miner_uid: Mapped[int] = mapped_column(Integer, nullable=False)
    code_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    bucket_path: Mapped[str] = mapped_column(String(1024), nullable=False)  # Code URL

    # Versioning - which validator version evaluated this
    spec_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, index=True)

    status: Mapped[SubmissionStatus] = mapped_column(
        Enum(SubmissionStatus, values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
        default=SubmissionStatus.PENDING,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # MFU is the primary metric (final_score stores MFU now, not TPS)
    final_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Miner's actual code (stored after evaluation for dashboard display)
    code_content: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Rate limiting handled by checking commit_block in submission_id
    payment_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Relationships
    evaluations: Mapped[list["EvaluationModel"]] = relationship(
        "EvaluationModel", back_populates="submission", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_submissions_status", "status"),
        Index("idx_submissions_final_score", "final_score"),
        Index("idx_submissions_created_at", "created_at"),
        Index("idx_submissions_spec_version", "spec_version"),
    )


class ValidatorStateModel(Base):
    """Database model for validator state persistence.

    Stores state that should survive validator restarts:
    - last_processed_block: To avoid re-processing old commitments
    - evaluated_code_urls: To avoid duplicate evaluations (stored as JSON)
    """

    __tablename__ = "validator_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class AdaptiveThresholdModel(Base):
    """Database model for adaptive leaderboard threshold.

    The threshold adapts based on improvement magnitude:
    - Big improvements (e.g., 24% → 48%) create a high threshold
    - Threshold decays over time towards base_threshold
    - This rewards big jumps and prevents gaming
    """

    __tablename__ = "adaptive_threshold"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    current_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.01)
    last_improvement: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_update_block: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class EvaluationModel(Base):
    """Database model for evaluation results."""

    __tablename__ = "evaluations"

    evaluation_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    submission_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("submissions.submission_id"), nullable=False
    )
    evaluator_hotkey: Mapped[str] = mapped_column(String(48), nullable=False, index=True)

    # MFU is the primary metric (Model FLOPs Utilization %)
    mfu: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    tokens_per_second: Mapped[float] = mapped_column(Float, nullable=False)  # Secondary metric
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    wall_time_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    submission: Mapped["SubmissionModel"] = relationship(
        "SubmissionModel", back_populates="evaluations"
    )

    __table_args__ = (
        Index("idx_evaluations_submission_id", "submission_id"),
        Index("idx_evaluations_evaluator", "evaluator_hotkey"),
        Index("idx_evaluations_mfu", "mfu"),
    )
