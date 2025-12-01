"""Leaderboard API endpoints."""

from fastapi import APIRouter, Depends

from tournament.schemas import LeaderboardEntry
from tournament.storage.database import Database, get_database

router = APIRouter(prefix="/leaderboard", tags=["leaderboard"])


@router.get("", response_model=list[LeaderboardEntry])
async def get_leaderboard(
    limit: int = 100,
    db: Database = Depends(get_database),
) -> list[LeaderboardEntry]:
    """Get the current leaderboard."""
    submissions = await db.get_leaderboard(limit=limit)

    return [
        LeaderboardEntry(
            rank=i + 1,
            submission_id=s.submission_id,
            miner_hotkey=s.miner_hotkey,
            miner_uid=s.miner_uid,
            final_score=s.final_score,
            num_evaluations=len(s.evaluations) if hasattr(s, "evaluations") else 0,
            created_at=s.created_at,
        )
        for i, s in enumerate(submissions)
    ]
