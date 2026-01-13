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
    """Database model for code submissions."""

    __tablename__ = "submissions"

    submission_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    miner_hotkey: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    miner_uid: Mapped[int] = mapped_column(Integer, nullable=False)
    code_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    bucket_path: Mapped[str] = mapped_column(String(256), nullable=False)

    status: Mapped[SubmissionStatus] = mapped_column(
        Enum(SubmissionStatus, values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
        default=SubmissionStatus.PENDING
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    final_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Payment fields (anti-spam mechanism)
    payment_block_hash: Mapped[str | None] = mapped_column(String(66), nullable=True)
    payment_extrinsic_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payment_amount_rao: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payment_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    
    # Anti-copying: Blockchain timestamp (proves code ownership)
    # Miner posts code_hash to chain BEFORE submitting to validator
    # Prevents malicious validators from stealing and resubmitting code
    code_timestamp_block_hash: Mapped[str | None] = mapped_column(String(66), nullable=True)
    code_timestamp_extrinsic_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Anti-copying: Structural fingerprint (for cross-validator copy detection)
    # Unlike code_hash which is completely different for any change,
    # fingerprint is similar for similar code structures
    # This allows validators to detect modified copies even if they never saw the original
    code_fingerprint: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

    # Relationships
    evaluations: Mapped[list["EvaluationModel"]] = relationship(
        "EvaluationModel", back_populates="submission", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_submissions_status", "status"),
        Index("idx_submissions_final_score", "final_score"),
        Index("idx_submissions_created_at", "created_at"),
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

    tokens_per_second: Mapped[float] = mapped_column(Float, nullable=False)
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
    )
