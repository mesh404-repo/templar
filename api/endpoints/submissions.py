"""Submission status API endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from tournament.schemas import EvaluationResponse, SubmissionResponse
from tournament.storage.database import Database, get_database

router = APIRouter(prefix="/submissions", tags=["submissions"])


@router.get("/{submission_id}", response_model=SubmissionResponse)
async def get_submission(
    submission_id: str,
    db: Database = Depends(get_database),
) -> SubmissionResponse:
    """Get submission status and details."""
    submission = await db.get_submission(submission_id)

    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    return SubmissionResponse.model_validate(submission)


@router.get("/{submission_id}/evaluations", response_model=list[EvaluationResponse])
async def get_submission_evaluations(
    submission_id: str,
    db: Database = Depends(get_database),
) -> list[EvaluationResponse]:
    """Get all evaluations for a submission."""
    submission = await db.get_submission(submission_id)

    if submission is None:
        raise HTTPException(status_code=404, detail="Submission not found")

    evaluations = await db.get_evaluations(submission_id)

    return [EvaluationResponse.model_validate(e) for e in evaluations]
