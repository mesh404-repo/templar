"""API client for crusades endpoints."""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from crusades.config import get_hparams
from crusades.core.protocols import SubmissionStatus


@dataclass
class CrusadesData:
    """Container for all crusades data."""

    overview: dict[str, Any]
    validator: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    recent: list[dict[str, Any]]
    queue: dict[str, Any]
    history: list[dict[str, Any]]
    threshold: dict[str, Any] | None = None  # Adaptive threshold info


@dataclass
class SubmissionDetail:
    """Container for submission detail data."""

    submission: dict[str, Any]
    evaluations: list[dict[str, Any]]
    code: str | None


class MockClient:
    """Mock client that returns demo data."""

    def __init__(self):
        from crusades.tui.mock_data import (
            MOCK_CODE,
            MOCK_EVALUATIONS,
            MOCK_HISTORY,
            MOCK_LEADERBOARD,
            MOCK_OVERVIEW,
            MOCK_QUEUE,
            MOCK_RECENT,
            MOCK_SUBMISSIONS,
            MOCK_VALIDATOR,
            get_default_evaluations,
            get_default_submission,
        )

        self._overview = MOCK_OVERVIEW
        self._validator = MOCK_VALIDATOR
        self._queue = MOCK_QUEUE
        self._leaderboard = MOCK_LEADERBOARD
        self._recent = MOCK_RECENT
        self._history = MOCK_HISTORY
        self._submissions = MOCK_SUBMISSIONS
        self._evaluations = MOCK_EVALUATIONS
        self._code = MOCK_CODE
        self._get_default_submission = get_default_submission
        self._get_default_evaluations = get_default_evaluations

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get_overview(self) -> dict[str, Any]:
        return self._overview

    def get_validator_status(self) -> dict[str, Any]:
        return self._validator

    def get_leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        return self._leaderboard[:limit]

    def get_recent_submissions(self) -> list[dict[str, Any]]:
        return self._recent

    def get_queue_stats(self) -> dict[str, Any]:
        return self._queue

    def get_history(self) -> list[dict[str, Any]]:
        return self._history

    def get_adaptive_threshold(self) -> dict[str, Any]:
        """Mock adaptive threshold."""
        return {
            "current_threshold": 0.20,
            "last_improvement": 0.20,
            "last_update_block": 1000,
            "decayed_threshold": 0.15,  # Decayed from 20% to 15%
        }

    def fetch_all(self) -> CrusadesData:
        return CrusadesData(
            overview=self.get_overview(),
            validator=self.get_validator_status(),
            leaderboard=self.get_leaderboard(),
            recent=self.get_recent_submissions(),
            queue=self.get_queue_stats(),
            history=self.get_history(),
            threshold=self.get_adaptive_threshold(),
        )

    def get_submission(self, submission_id: str) -> dict[str, Any]:
        return self._submissions.get(submission_id, self._get_default_submission(submission_id))

    def get_submission_evaluations(self, submission_id: str) -> list[dict[str, Any]]:
        return self._evaluations.get(submission_id, self._get_default_evaluations(submission_id))

    def get_submission_code(self, submission_id: str) -> str | None:
        return self._code

    def fetch_submission_detail(self, submission_id: str) -> SubmissionDetail:
        return SubmissionDetail(
            submission=self.get_submission(submission_id),
            evaluations=self.get_submission_evaluations(submission_id),
            code=self.get_submission_code(submission_id),
        )


class DatabaseClient:
    """Client that reads directly from validator's SQLite database."""

    def __init__(self, db_path: str = "crusades.db", spec_version: int | None = None):
        self.db_path = Path(db_path)
        self.spec_version = spec_version
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def _version_filter(self, alias: str = "s") -> str:
        """Get SQL WHERE clause for spec_version filtering.

        Note: spec_version is validated as int in __init__ and derived from
        hardcoded package version, not user input. Using int() as defense-in-depth.
        """
        if self.spec_version is None:
            return ""
        # Validate spec_version is an integer to prevent SQL injection
        version = int(self.spec_version)
        return f" AND {alias}.spec_version = {version}"

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute query and return list of dicts."""
        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def _query_one(self, sql: str, params: tuple = ()) -> dict | None:
        """Execute query and return single dict or None."""
        cursor = self._conn.execute(sql, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_adaptive_threshold(
        self,
        base_threshold: float | None = None,
        decay_percent: float | None = None,
        decay_interval_blocks: int | None = None,
        block_time: int | None = None,
    ) -> dict[str, Any]:
        """Get current adaptive threshold info.

        Calculates the decayed threshold based on time elapsed since last update.
        Uses values from hparams.json if not explicitly provided.
        """
        # Get defaults from hparams
        hparams = get_hparams()
        threshold_config = hparams.adaptive_threshold
        base_threshold = base_threshold if base_threshold is not None else threshold_config.base_threshold
        decay_percent = decay_percent if decay_percent is not None else threshold_config.decay_percent
        decay_interval_blocks = decay_interval_blocks if decay_interval_blocks is not None else threshold_config.decay_interval_blocks
        block_time = block_time if block_time is not None else hparams.block_time
        try:
            row = self._query_one(
                "SELECT current_threshold, last_improvement, last_update_block, updated_at "
                "FROM adaptive_threshold WHERE id = 1"
            )
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            row = None

        if not row:
            return {
                "current_threshold": base_threshold,
                "last_improvement": 0.0,
                "last_update_block": 0,
                "decayed_threshold": base_threshold,
            }

        # Calculate decay based on time elapsed (estimate blocks from time)
        current_threshold = row["current_threshold"]
        updated_at_str = row["updated_at"]

        # Parse updated_at and calculate elapsed time
        try:
            updated_at_str = (
                updated_at_str.replace(" ", "T") if " " in updated_at_str else updated_at_str
            )
            # Handle timezone-aware datetimes (e.g., "2026-01-07T10:35:00Z")
            if updated_at_str.endswith("Z"):
                updated_at_str = updated_at_str[:-1] + "+00:00"
            updated_at = datetime.fromisoformat(updated_at_str)

            # Use timezone-aware now() if updated_at is tz-aware, else naive
            if updated_at.tzinfo is not None:
                now = datetime.now(UTC)
            else:
                now = datetime.now()

            elapsed_seconds = (now - updated_at).total_seconds()
            # Clamp to non-negative to prevent decay_factor > 1.0 from clock skew
            elapsed_seconds = max(0, elapsed_seconds)
            
            # Guard against misconfigured intervals (avoid division by zero)
            if block_time <= 0 or decay_interval_blocks <= 0:
                decayed = current_threshold
            else:
                elapsed_blocks = elapsed_seconds / block_time
                decay_steps = elapsed_blocks / decay_interval_blocks
                decay_factor = (1.0 - decay_percent) ** decay_steps
                decayed = base_threshold + (current_threshold - base_threshold) * decay_factor
        except (ValueError, TypeError):
            decayed = current_threshold

        return {
            "current_threshold": current_threshold,
            "last_improvement": row["last_improvement"],
            "last_update_block": row["last_update_block"],
            "decayed_threshold": max(base_threshold, decayed),
            "updated_at": row["updated_at"],
        }

    def get_threshold_winner(self, threshold: float = 0.01) -> dict[str, Any] | None:
        """Get the threshold-adjusted winner (for weight distribution).

        A new submission only beats an incumbent if it's more than `threshold` better.
        This gives stability to leaders and matches the weight setting logic.
        """
        vf = self._version_filter("s")
        # Get all finished submissions ordered by created_at (oldest first)
        rows = self._query(
            f"""SELECT submission_id, final_score, created_at
               FROM submissions s
               WHERE status = ? AND final_score IS NOT NULL{vf}
               ORDER BY created_at ASC""",
            (SubmissionStatus.FINISHED,),
        )

        if not rows:
            return None

        # Build leaderboard with threshold logic (same as database.py get_leaderboard_winner)
        leaderboard: list[dict] = []
        for row in rows:
            score = row["final_score"] or 0.0
            insert_pos = len(leaderboard)

            for i, existing in enumerate(leaderboard):
                existing_score = existing["final_score"] or 0.0
                threshold_score = existing_score * (1 + threshold)
                if score > threshold_score:
                    insert_pos = i
                    break

            leaderboard.insert(insert_pos, dict(row))

        return leaderboard[0] if leaderboard else None

    def get_overview(self) -> dict[str, Any]:
        """Get dashboard overview stats (filtered by spec_version)."""
        now = datetime.now()
        day_ago = (now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        vf = self._version_filter("s")

        # Total submissions
        total = self._query_one(f"SELECT COUNT(*) as count FROM submissions s WHERE 1=1{vf}")
        total_count = total["count"] if total else 0

        # Submissions in last 24h
        recent = self._query_one(
            f"SELECT COUNT(*) as count FROM submissions s WHERE created_at > ?{vf}",
            (day_ago,),
        )
        recent_count = recent["count"] if recent else 0

        # Get adaptive threshold for calculations
        threshold_data = self.get_adaptive_threshold()
        adaptive_threshold = threshold_data.get("decayed_threshold", 0.01)

        # Top MFU (threshold-adjusted winner - the one who gets weights)
        threshold_winner = self.get_threshold_winner(threshold=adaptive_threshold)
        top_score = threshold_winner["final_score"] if threshold_winner else 0.0

        # Top score from 24 hours ago
        top_24h_ago = self._query_one(
            f"SELECT MAX(final_score) as score FROM submissions s "
            f"WHERE status = 'finished' AND created_at <= ?{vf}",
            (day_ago,),
        )
        score_24h_ago = top_24h_ago["score"] if top_24h_ago and top_24h_ago["score"] else 0.0

        # Calculate improvement percentage (never negative)
        if score_24h_ago > 0:
            improvement = max(0, ((top_score - score_24h_ago) / score_24h_ago) * 100)
        elif top_score > 0:
            first_score = self._query_one(
                f"SELECT final_score FROM submissions s WHERE status = 'finished' "
                f"AND final_score > 0{vf} ORDER BY created_at ASC LIMIT 1"
            )
            baseline = (
                first_score["final_score"] if first_score and first_score["final_score"] else 0.0
            )
            if baseline > 0 and baseline != top_score:
                improvement = max(0, ((top_score - baseline) / baseline) * 100)
            else:
                improvement = 0.0
        else:
            improvement = 0.0

        # Active miners (unique hotkeys in last 24h)
        miners = self._query_one(
            f"SELECT COUNT(DISTINCT miner_hotkey) as count FROM submissions s "
            f"WHERE created_at > ?{vf}",
            (day_ago,),
        )
        active_miners = miners["count"] if miners else 0

        # MFU to Beat = threshold winner's score * (1 + threshold)
        mfu_to_beat = top_score * (1 + adaptive_threshold) if top_score > 0 else 0.0

        return {
            "submissions_24h": recent_count,
            "current_top_score": top_score,
            "mfu_to_beat": round(mfu_to_beat, 4),
            "adaptive_threshold": adaptive_threshold,
            "score_improvement_24h": round(improvement, 2),
            "total_submissions": total_count,
            "active_miners": active_miners,
        }

    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string."""
        if seconds < 0:
            return "N/A"

        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

    def get_validator_status(self) -> dict[str, Any]:
        """Get validator status (filtered by spec_version)."""
        now = datetime.now()
        hour_ago = (now - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        vf = self._version_filter("s")

        # Evaluations in last hour (join with submissions for version filter)
        evals = self._query_one(
            f"SELECT COUNT(*) as count FROM evaluations e "
            f"JOIN submissions s ON e.submission_id = s.submission_id "
            f"WHERE e.created_at > ?{vf}",
            (hour_ago,),
        )
        eval_count = evals["count"] if evals else 0

        # Current evaluation (most recent evaluating submission)
        current = self._query_one(
            f"SELECT submission_id FROM submissions s WHERE status = 'evaluating'{vf} "
            f"ORDER BY created_at DESC LIMIT 1"
        )

        # Queue stats
        queue = self.get_queue_stats()

        # Success rate
        finished = self._query_one(
            f"SELECT COUNT(*) as count FROM submissions s WHERE status = 'finished'{vf}"
        )
        failed = self._query_one(
            f"SELECT COUNT(*) as count FROM submissions s WHERE status LIKE 'failed%'{vf}"
        )
        finished_count = finished["count"] if finished else 0
        failed_count = failed["count"] if failed else 0
        total = finished_count + failed_count
        success_rate = (finished_count / total * 100) if total > 0 else 0

        # Calculate uptime from first submission
        first_submission = self._query_one(
            f"SELECT MIN(created_at) as start_time FROM submissions s WHERE 1=1{vf}"
        )
        if first_submission and first_submission["start_time"]:
            try:
                start_str = first_submission["start_time"]
                start_str = start_str.replace(" ", "T")
                start_time = datetime.fromisoformat(start_str)
                uptime_seconds = (now - start_time).total_seconds()
                uptime = self._format_duration(uptime_seconds)
            except (ValueError, TypeError):
                uptime = "N/A"
        else:
            uptime = "N/A"

        return {
            "status": "running" if current else "idle",
            "evaluations_completed_1h": eval_count,
            "current_evaluation": current["submission_id"] if current else None,
            "uptime": uptime,
            "queued_count": queue["queued_count"],
            "running_count": queue["running_count"],
            "finished_count": queue["finished_count"],
            "failed_count": queue["failed_count"],
            "success_rate": f"{success_rate:.1f}%",
        }

    def get_leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get leaderboard with threshold winner at #1, rest sorted by raw MFU.

        Position #1: Threshold-adjusted winner (gets emissions)
        Positions #2+: All others sorted by raw MFU descending
        """
        # Get adaptive threshold
        threshold_data = self.get_adaptive_threshold()
        threshold = threshold_data.get("decayed_threshold", 0.01)

        # Get threshold winner (position #1)
        winner = self.get_threshold_winner(threshold=threshold)
        winner_id = winner["submission_id"] if winner else None

        # Get all submissions sorted by raw score
        vf = self._version_filter("s")
        rows = self._query(
            f"""SELECT s.submission_id, s.miner_hotkey, s.miner_uid, s.final_score,
                      s.created_at, COUNT(e.evaluation_id) as eval_count
               FROM submissions s
               LEFT JOIN evaluations e ON s.submission_id = e.submission_id
               WHERE s.status = ? AND s.final_score IS NOT NULL{vf}
               GROUP BY s.submission_id
               ORDER BY s.final_score DESC""",
            (SubmissionStatus.FINISHED,),
        )

        # Build leaderboard: winner first, then others by raw score
        leaderboard = []

        # Add winner as #1 (if exists)
        if winner:
            # Find winner's full data from rows
            winner_row = next((r for r in rows if r["submission_id"] == winner_id), None)
            if winner_row:
                leaderboard.append({
                    "rank": 1,
                    "submission_id": winner_row["submission_id"],
                    "miner_hotkey": winner_row["miner_hotkey"],
                    "miner_uid": winner_row["miner_uid"],
                    "final_score": winner_row["final_score"] or 0.0,
                    "num_evaluations": winner_row["eval_count"],
                    "created_at": winner_row["created_at"],
                })

        # Add remaining submissions (excluding winner) sorted by raw score
        rank = 2 if winner else 1
        for row in rows:
            if row["submission_id"] != winner_id:
                leaderboard.append({
                    "rank": rank,
                    "submission_id": row["submission_id"],
                    "miner_hotkey": row["miner_hotkey"],
                    "miner_uid": row["miner_uid"],
                    "final_score": row["final_score"] or 0.0,
                    "num_evaluations": row["eval_count"],
                    "created_at": row["created_at"],
                })
                rank += 1
                if len(leaderboard) >= limit:
                    break

        return leaderboard[:limit]

    def get_recent_submissions(self) -> list[dict[str, Any]]:
        """Get recent submissions (filtered by spec_version)."""
        vf = self._version_filter("s")
        rows = self._query(
            f"SELECT submission_id, miner_hotkey, miner_uid, status, final_score, "
            f"created_at, error_message FROM submissions s "
            f"WHERE status IN ('finished', 'failed_validation', 'failed_evaluation', "
            f"'error'){vf} "
            f"ORDER BY created_at DESC LIMIT 20"
        )

        recent = []
        for row in rows:
            recent.append(
                {
                    "submission_id": row["submission_id"],
                    "miner_hotkey": row["miner_hotkey"],
                    "miner_uid": row["miner_uid"],
                    "status": row["status"],
                    "final_score": row["final_score"],  # Match app.py expectation
                    "created_at": row["created_at"],
                    "error": row["error_message"],
                }
            )
        return recent

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics (filtered by spec_version)."""
        vf = self._version_filter("s")

        queued = self._query_one(
            f"SELECT COUNT(*) as count FROM submissions s "
            f"WHERE status IN ('pending', 'validating'){vf}"
        )

        running = self._query_one(
            f"SELECT COUNT(*) as count FROM submissions s WHERE status = 'evaluating'{vf}"
        )

        finished = self._query_one(
            f"SELECT COUNT(*) as count FROM submissions s WHERE status = 'finished'{vf}"
        )

        failed = self._query_one(
            f"SELECT COUNT(*) as count FROM submissions s WHERE status IN (?, ?, ?){vf}",
            (
                SubmissionStatus.FAILED_VALIDATION,
                SubmissionStatus.FAILED_EVALUATION,
                SubmissionStatus.ERROR,
            ),
        )

        avg = self._query_one(
            f"SELECT AVG(final_score) as avg FROM submissions s "
            f"WHERE status = 'finished' AND final_score IS NOT NULL{vf}"
        )

        return {
            "queued_count": queued["count"] if queued else 0,
            "running_count": running["count"] if running else 0,
            "finished_count": finished["count"] if finished else 0,
            "failed_count": failed["count"] if failed else 0,
            "avg_wait_time_seconds": 0.0,
            "avg_score": avg["avg"] if avg and avg["avg"] else 0.0,
            "pending_count": queued["count"] if queued else 0,
        }

    def get_history(self) -> list[dict[str, Any]]:
        """Get MFU history for chart (filtered by spec_version).

        Shows running maximum MFU over time (simple max, no threshold).
        """
        vf = self._version_filter("s")
        rows = self._query(
            f"SELECT submission_id, final_score as mfu, created_at "
            f"FROM submissions s "
            f"WHERE status = 'finished' AND final_score IS NOT NULL{vf} "
            f"ORDER BY created_at ASC"
        )

        history = []
        running_best = 0.0

        for row in rows:
            mfu = row["mfu"] or 0.0
            if mfu > running_best:
                running_best = mfu

            history.append(
                {
                    "submission_id": row["submission_id"],
                    "mfu": running_best,
                    "timestamp": row["created_at"],
                }
            )
        return history

    def fetch_all(self) -> CrusadesData:
        """Fetch all crusades data."""
        return CrusadesData(
            overview=self.get_overview(),
            validator=self.get_validator_status(),
            leaderboard=self.get_leaderboard(),
            recent=self.get_recent_submissions(),
            queue=self.get_queue_stats(),
            history=self.get_history(),
            threshold=self.get_adaptive_threshold(),
        )

    def get_submission(self, submission_id: str) -> dict[str, Any]:
        """Get submission details."""
        row = self._query_one("SELECT * FROM submissions WHERE submission_id = ?", (submission_id,))
        return row if row else {}

    def get_submission_evaluations(self, submission_id: str) -> list[dict[str, Any]]:
        """Get evaluations for a submission."""
        return self._query(
            "SELECT * FROM evaluations WHERE submission_id = ? ORDER BY created_at",
            (submission_id,),
        )

    def get_submission_code(self, submission_id: str) -> str | None:
        """Get code content from database (stored after evaluation).

        Code is stored in code_content column after validator evaluates.
        """
        row = self._query_one(
            "SELECT code_content, code_hash FROM submissions WHERE submission_id = ?",
            (submission_id,),
        )
        if not row:
            return None

        code = row.get("code_content")
        if code:
            return code

        # Fallback: code not yet stored (still evaluating)
        return f"# Code not yet available\n# Submission may still be evaluating\n# code_hash: {row.get('code_hash', 'N/A')}"

    def fetch_submission_detail(self, submission_id: str) -> SubmissionDetail:
        """Fetch all details for a submission."""
        return SubmissionDetail(
            submission=self.get_submission(submission_id),
            evaluations=self.get_submission_evaluations(submission_id),
            code=self.get_submission_code(submission_id),
        )


class CrusadesClient:
    """Client for fetching crusades data from API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=10.0)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _get(self, endpoint: str) -> dict[str, Any] | list[Any]:
        """Make a GET request to the API."""
        try:
            response = self._client.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return {}

    def get_overview(self) -> dict[str, Any]:
        """Get dashboard overview stats."""
        data = self._get("/api/stats/overview")
        if not data:
            return {
                "submissions_24h": 0,
                "current_top_score": 0.0,
                "score_improvement_24h": 0.0,
                "total_submissions": 0,
                "active_miners": 0,
            }
        return data

    def get_validator_status(self) -> dict[str, Any]:
        """Get validator status."""
        data = self._get("/api/stats/validator")
        if not data:
            return {
                "status": "unknown",
                "evaluations_completed_1h": 0,
                "current_evaluation": None,
                "uptime": "N/A",
            }
        return data

    def get_leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get leaderboard entries."""
        data = self._get(f"/leaderboard?limit={limit}")
        if not data or not isinstance(data, list):
            return []
        return data

    def get_recent_submissions(self) -> list[dict[str, Any]]:
        """Get recent submissions."""
        data = self._get("/api/stats/recent")
        if not data or not isinstance(data, list):
            return []
        return data

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        data = self._get("/api/stats/queue")
        if not data:
            return {
                "queued_count": 0,
                "running_count": 0,
                "finished_count": 0,
                "failed_count": 0,
                "avg_wait_time_seconds": 0.0,
                "avg_score": 0.0,
                "pending_count": 0,
            }
        return data

    def get_history(self) -> list[dict[str, Any]]:
        """Get MFU history."""
        data = self._get("/api/stats/history")
        if not data or not isinstance(data, list):
            return []
        return data

    def get_adaptive_threshold(self) -> dict[str, Any]:
        """Get current adaptive threshold."""
        data = self._get("/api/stats/threshold")
        if not data:
            return {
                "current_threshold": 0.01,
                "last_improvement": 0.0,
                "last_update_block": 0,
                "decayed_threshold": 0.01,
            }
        return data

    def fetch_all(self) -> CrusadesData:
        """Fetch all crusades data."""
        return CrusadesData(
            overview=self.get_overview(),
            validator=self.get_validator_status(),
            leaderboard=self.get_leaderboard(),
            recent=self.get_recent_submissions(),
            queue=self.get_queue_stats(),
            history=self.get_history(),
            threshold=self.get_adaptive_threshold(),
        )

    def get_submission(self, submission_id: str) -> dict[str, Any]:
        """Get submission details."""
        data = self._get(f"/api/submissions/{submission_id}")
        if not data:
            return {}
        return data

    def get_submission_evaluations(self, submission_id: str) -> list[dict[str, Any]]:
        """Get evaluations for a submission."""
        data = self._get(f"/api/submissions/{submission_id}/evaluations")
        if not data or not isinstance(data, list):
            return []
        return data

    def get_submission_code(self, submission_id: str) -> str | None:
        """Get code for a submission."""
        data = self._get(f"/api/submissions/{submission_id}/code")
        if not data or not isinstance(data, dict):
            return None
        return data.get("code")

    def fetch_submission_detail(self, submission_id: str) -> SubmissionDetail:
        """Fetch all details for a submission."""
        return SubmissionDetail(
            submission=self.get_submission(submission_id),
            evaluations=self.get_submission_evaluations(submission_id),
            code=self.get_submission_code(submission_id),
        )


def format_time_ago(timestamp_str: str | None) -> str:
    """Format a timestamp as 'X ago' string."""
    if not timestamp_str:
        return "N/A"
    try:
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(timestamp_str)
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = now - dt

        if delta.days > 0:
            return f"{delta.days}d ago"
        hours = delta.seconds // 3600
        if hours > 0:
            return f"{hours}h ago"
        minutes = delta.seconds // 60
        if minutes > 0:
            return f"{minutes}m ago"
        return "just now"
    except (ValueError, TypeError):
        return "N/A"
