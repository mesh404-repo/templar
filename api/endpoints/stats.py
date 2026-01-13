"""Statistics API endpoints for dashboard."""

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends

from tournament.core.protocols import SubmissionStatus
from tournament.storage.database import Database, get_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/validator")
async def get_validator_status(db: Database = Depends(get_database)) -> dict:
    """Get current validator status."""
    try:
        # Get recent evaluations to show validator activity
        all_submissions = await db.get_all_submissions()
        
        # Count evaluations in last hour
        now = datetime.utcnow()
        last_hour = now - timedelta(hours=1)
        
        recent_evals = []
        for s in all_submissions:
            if s.created_at > last_hour:
                evals = await db.get_evaluations(s.submission_id)
                recent_evals.extend(evals)
        
        # Find currently evaluating submission
        evaluating = await db.get_evaluating_submissions()
        current_eval = evaluating[0] if evaluating else None
        
        return {
            "status": "running",
            "evaluations_completed_1h": len(recent_evals),
            "current_evaluation": {
                "submission_id": current_eval.submission_id if current_eval else None,
                "miner_uid": current_eval.miner_uid if current_eval else None,
            } if current_eval else None,
            "uptime": "Running",
        }
    except Exception as e:
        return {
            "status": "unknown",
            "evaluations_completed_1h": 0,
            "current_evaluation": None,
            "uptime": "Unknown",
        }


@router.get("/history")
async def get_tps_history(
    limit: int = 50,
    db: Database = Depends(get_database),
) -> list[dict]:
    """Get historical TPS data for charting."""
    try:
        submissions = await db.get_leaderboard(limit=limit)
        
        # Return time series data
        return [
            {
                "timestamp": s.created_at.isoformat(),
                "tps": s.final_score,
                "miner_uid": s.miner_uid,
                "submission_id": s.submission_id,
            }
            for s in reversed(submissions)  # Oldest first for chart
        ]
    except Exception as e:
        return []


@router.get("/overview")
async def get_overview_stats(
    db: Database = Depends(get_database),
) -> dict:
    """Get overview statistics for dashboard.
    
    Returns metrics similar to Ridges.ai:
    - Total submissions (last 24h)
    - Current top score
    - Score improvement (24h)
    - Active validators
    """
    try:
        # Time boundaries
        now = datetime.utcnow()
        yesterday = now - timedelta(days=1)
        
        # Get all submissions safely
        try:
            all_submissions = await db.get_all_submissions()
        except Exception as e:
            # If get_all_submissions fails, try getting from different queries
            logger.warning(f"get_all_submissions failed: {e}, trying alternative")
            pending = await db.get_pending_submissions()
            evaluating = await db.get_evaluating_submissions()
            leaderboard = await db.get_leaderboard(limit=100)
            all_submissions = pending + evaluating + leaderboard
        
        # Submissions in last 24h
        recent_submissions = [
            s for s in all_submissions 
            if s.created_at >= yesterday
        ]
        submissions_24h = len(recent_submissions)
        
        # Current top score from all submissions
        all_scores = [s.final_score for s in all_submissions if s.final_score is not None]
        current_top_score = max(all_scores) if all_scores else 0.0
        
        # Top score 24h ago
        old_submissions = [
            s for s in all_submissions
            if s.created_at < yesterday and s.final_score is not None
        ]
        old_top_score = max([s.final_score for s in old_submissions], default=0.0)
        
        # Score improvement
        score_improvement = current_top_score - old_top_score if old_top_score > 0 else 0.0
        improvement_pct = (score_improvement / old_top_score * 100) if old_top_score > 0 else 0.0
        
        return {
            "submissions_24h": submissions_24h,
            "current_top_score": current_top_score,
            "score_improvement_24h": improvement_pct,
            "total_submissions": len(all_submissions),
            "active_miners": len(set(s.miner_uid for s in all_submissions)),
        }
    except Exception as e:
        # Return zeros if database has issues
        return {
            "submissions_24h": 0,
            "current_top_score": 0.0,
            "score_improvement_24h": 0.0,
            "total_submissions": 0,
            "active_miners": 0,
        }


@router.get("/queue")
async def get_queue_stats(
    db: Database = Depends(get_database),
) -> dict:
    """Get queue statistics.
    
    Returns:
    - Pending submissions count
    - Running submissions count
    - Average wait time
    - Average score
    """
    try:
        all_submissions = await db.get_all_submissions()
        
        # Count by status
        pending = [s for s in all_submissions if s.status == SubmissionStatus.PENDING]
        running = [s for s in all_submissions if s.status == SubmissionStatus.RUNNING]
        finished = [s for s in all_submissions if s.status == SubmissionStatus.FINISHED]
        
        # Calculate average wait time for finished submissions
        wait_times = []
        for s in finished:
            if s.created_at and s.updated_at:
                wait_time = (s.updated_at - s.created_at).total_seconds()
                wait_times.append(wait_time)
        
        avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0
        
        # Average score of finished submissions
        scores = [s.final_score for s in finished if s.final_score is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "pending_count": len(pending),
            "running_count": len(running),
            "finished_count": len(finished),
            "avg_wait_time_seconds": avg_wait_time,
            "avg_score": avg_score,
        }
    except Exception as e:
        return {
            "pending_count": 0,
            "running_count": 0,
            "finished_count": 0,
            "avg_wait_time_seconds": 0,
            "avg_score": 0,
        }


@router.get("/recent")
async def get_recent_submissions(
    limit: int = 50,
    db: Database = Depends(get_database),
) -> list[dict]:
    """Get recent submissions with details for activity feed.
    
    SECURITY: Only show finished/failed submissions to prevent code theft.
    Pending/evaluating submissions are hidden until complete.
    """
    try:
        submissions = await db.get_all_submissions()
        
        # SECURITY: Filter out pending/evaluating submissions
        # This prevents attackers from seeing submissions before they're safe
        visible_statuses = ['finished', 'failed_validation', 'error']
        visible_submissions = [
            s for s in submissions 
            if s.status.value in visible_statuses
        ]
        
        # Sort by created_at descending
        visible_submissions.sort(key=lambda s: s.created_at, reverse=True)
        
        return [
            {
                "submission_id": s.submission_id,
                "miner_uid": s.miner_uid,
                "miner_hotkey": s.miner_hotkey[:8] + "...",  # Truncate for privacy
                "status": s.status.value,
                "final_score": s.final_score if s.final_score else 0,
                "created_at": s.created_at.isoformat(),
            }
            for s in visible_submissions[:limit]
        ]
    except Exception as e:
        return []



