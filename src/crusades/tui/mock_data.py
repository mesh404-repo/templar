"""Mock data for TUI demo mode."""

# MFU (Model FLOPs Utilization) shown as percentage (0-100%)
MOCK_HISTORY = [
    {"timestamp": "2026-01-05T10:00:00Z", "mfu": 28.5, "miner_uid": 41, "submission_id": "sub_012"},
    {"timestamp": "2026-01-05T14:00:00Z", "mfu": 29.2, "miner_uid": 9, "submission_id": "sub_011"},
    {"timestamp": "2026-01-05T18:00:00Z", "mfu": 30.5, "miner_uid": 27, "submission_id": "sub_010"},
    {"timestamp": "2026-01-05T22:00:00Z", "mfu": 31.8, "miner_uid": 14, "submission_id": "sub_009"},
    {"timestamp": "2026-01-06T10:00:00Z", "mfu": 33.1, "miner_uid": 52, "submission_id": "sub_008"},
    {"timestamp": "2026-01-06T14:00:00Z", "mfu": 34.5, "miner_uid": 19, "submission_id": "sub_007"},
    {"timestamp": "2026-01-06T18:00:00Z", "mfu": 36.0, "miner_uid": 31, "submission_id": "sub_006"},
    {"timestamp": "2026-01-06T20:00:00Z", "mfu": 37.1, "miner_uid": 8, "submission_id": "sub_005"},
    {"timestamp": "2026-01-06T22:00:00Z", "mfu": 38.5, "miner_uid": 45, "submission_id": "sub_004"},
    {"timestamp": "2026-01-07T08:00:00Z", "mfu": 39.2, "miner_uid": 23, "submission_id": "sub_003"},
    {"timestamp": "2026-01-07T09:00:00Z", "mfu": 40.9, "miner_uid": 7, "submission_id": "sub_002"},
    {"timestamp": "2026-01-07T10:00:00Z", "mfu": 43.0, "miner_uid": 12, "submission_id": "sub_001"},
]

MOCK_OVERVIEW = {
    "submissions_24h": 47,
    "current_top_score": 43.0,  # MFU percentage
    "mfu_to_beat": 43.43,  # top_score * (1 + adaptive_threshold)
    "adaptive_threshold": 0.01,  # 1% threshold
    "score_improvement_24h": 12.5,
    "total_submissions": 1283,
    "active_miners": 23,
}

MOCK_VALIDATOR = {
    "status": "running",
    "evaluations_completed_1h": 18,
    "current_evaluation": {
        "submission_id": "sub_abc123",
        "miner_uid": 42,
    },
    "uptime": "96.2%",
}

MOCK_QUEUE = {
    "queued_count": 3,
    "running_count": 1,
    "finished_count": 1279,
    "failed_count": 47,
    "avg_wait_time_seconds": 45.2,
    "avg_score": 38.4,  # MFU % average
    # Backwards compatibility
    "pending_count": 3,
}

MOCK_LEADERBOARD = [
    {
        "rank": 1,
        "submission_id": "sub_001",
        "miner_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "miner_uid": 12,
        "final_score": 43.0,  # MFU %
        "num_evaluations": 3,
        "created_at": "2026-01-07T10:30:00Z",
    },
    {
        "rank": 2,
        "submission_id": "sub_002",
        "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "miner_uid": 7,
        "final_score": 41.6,  # MFU %
        "num_evaluations": 3,
        "created_at": "2026-01-07T09:15:00Z",
    },
    {
        "rank": 3,
        "submission_id": "sub_003",
        "miner_hotkey": "5DAAnrj7VHTznn2AWBemMuyBwZWs6FNFjdyVXUeYum3PTXFy",
        "miner_uid": 23,
        "final_score": 40.9,  # MFU %
        "num_evaluations": 3,
        "created_at": "2026-01-07T08:45:00Z",
    },
    {
        "rank": 4,
        "submission_id": "sub_004",
        "miner_hotkey": "5HGjWAeFDfFCWPsjFQdVV2Msvz2XtMktvgocEZcCj68kUMaw",
        "miner_uid": 45,
        "final_score": 39.2,  # MFU %
        "num_evaluations": 3,
        "created_at": "2026-01-06T22:10:00Z",
    },
    {
        "rank": 5,
        "submission_id": "sub_005",
        "miner_hotkey": "5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL",
        "miner_uid": 8,
        "final_score": 38.5,  # MFU %
        "num_evaluations": 3,
        "created_at": "2026-01-06T20:30:00Z",
    },
    {
        "rank": 6,
        "submission_id": "sub_006",
        "miner_hotkey": "5GNJqTPyNqANBkUVMN1LPPrxXnFouWXoe2wNSmmEoLctxiZY",
        "miner_uid": 31,
        "final_score": 37.1,  # MFU %
        "num_evaluations": 3,
        "created_at": "2026-01-06T18:00:00Z",
    },
    {
        "rank": 7,
        "submission_id": "sub_007",
        "miner_hotkey": "5HpG9w8EBLe5XCrbczpwq5TSXvedjrBGCwqxK1iQ7qUsSWFc",
        "miner_uid": 19,
        "final_score": 36.5,  # MFU %
        "num_evaluations": 3,
        "created_at": "2026-01-06T15:45:00Z",
    },
    {
        "rank": 8,
        "submission_id": "sub_008",
        "miner_hotkey": "5Ck5SLSHYac6WFt5UZRSsdJjwmpSZq85fd5TRNAdZQVzEAPT",
        "miner_uid": 52,
        "final_score": 36.0,  # MFU %
        "num_evaluations": 3,
        "created_at": "2026-01-06T14:20:00Z",
    },
    {
        "rank": 9,
        "submission_id": "sub_009",
        "miner_hotkey": "5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y",
        "miner_uid": 3,
        "final_score": 35.2,  # MFU %
        "num_evaluations": 3,
        "created_at": "2026-01-06T12:00:00Z",
    },
    {
        "rank": 10,
        "submission_id": "sub_010",
        "miner_hotkey": "5HKPmK9GYtE1PSLsS1p9Dy8EFvLWDvFJLJNxNH8vqPqXq3UY",
        "miner_uid": 67,
        "final_score": 35.0,  # MFU %
        "num_evaluations": 3,
        "created_at": "2026-01-06T10:30:00Z",
    },
]

MOCK_RECENT = [
    {
        "submission_id": "sub_new_001",
        "miner_uid": 42,
        "miner_hotkey": "5GrwvaEF...",
        "status": "evaluating",
        "final_score": None,
        "created_at": "2026-01-07T11:45:00Z",
    },
    {
        "submission_id": "sub_new_002",
        "miner_uid": 15,
        "miner_hotkey": "5FHneW46...",
        "status": "validating",
        "final_score": None,
        "created_at": "2026-01-07T11:40:00Z",
    },
    {
        "submission_id": "sub_pending",
        "miner_uid": 28,
        "miner_hotkey": "5HpG9w8E...",
        "status": "pending",
        "final_score": None,
        "created_at": "2026-01-07T11:50:00Z",
    },
    {
        "submission_id": "sub_001",
        "miner_uid": 12,
        "miner_hotkey": "5DAAnrj7...",
        "status": "finished",
        "final_score": 43.0,  # MFU %
        "created_at": "2026-01-07T10:30:00Z",
    },
    {
        "submission_id": "sub_002",
        "miner_uid": 7,
        "miner_hotkey": "5HGjWAeF...",
        "status": "finished",
        "final_score": 41.6,  # MFU %
        "created_at": "2026-01-07T09:15:00Z",
    },
    {
        "submission_id": "sub_fail_001",
        "miner_uid": 33,
        "miner_hotkey": "5CiPPseX...",
        "status": "failed_validation",
        "final_score": None,
        "created_at": "2026-01-07T08:50:00Z",
    },
    {
        "submission_id": "sub_fail_eval",
        "miner_uid": 55,
        "miner_hotkey": "5FailEval...",
        "status": "failed_evaluation",
        "final_score": None,
        "created_at": "2026-01-07T08:40:00Z",
    },
    {
        "submission_id": "sub_timeout",
        "miner_uid": 77,
        "miner_hotkey": "5TimeoutHk...",
        "status": "failed_evaluation",
        "final_score": None,
        "created_at": "2026-01-07T08:35:00Z",
    },
    {
        "submission_id": "sub_003",
        "miner_uid": 23,
        "miner_hotkey": "5GNJqTPy...",
        "status": "finished",
        "final_score": 4089,
        "created_at": "2026-01-07T08:45:00Z",
    },
    {
        "submission_id": "sub_error",
        "miner_uid": 99,
        "miner_hotkey": "5ErrorHk...",
        "status": "error",
        "final_score": None,
        "created_at": "2026-01-07T07:30:00Z",
    },
]

# Submission details keyed by submission_id
MOCK_SUBMISSIONS = {
    "sub_001": {
        "submission_id": "sub_001",
        "miner_hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
        "miner_uid": 12,
        "code_hash": "a1b2c3d4e5f6789012345678901234567890abcdef",
        "status": "finished",
        "created_at": "2026-01-07T10:30:00Z",
        "updated_at": "2026-01-07T10:35:00Z",
        "final_score": 43.0,  # MFU %
        "error_message": None,
    },
    "sub_002": {
        "submission_id": "sub_002",
        "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
        "miner_uid": 7,
        "code_hash": "b2c3d4e5f67890123456789012345678abcdef01",
        "status": "finished",
        "created_at": "2026-01-07T09:15:00Z",
        "updated_at": "2026-01-07T09:20:00Z",
        "final_score": 41.6,  # MFU %
        "error_message": None,
    },
    "sub_fail_001": {
        "submission_id": "sub_fail_001",
        "miner_hotkey": "5CiPPseXPECbkjWCa6MnjNokrgYjMqmKndv2rSnekmSK2DjL",
        "miner_uid": 33,
        "code_hash": "fail123456789012345678901234567890abcdef",
        "status": "failed_validation",
        "created_at": "2026-01-07T08:50:00Z",
        "updated_at": "2026-01-07T08:51:00Z",
        "final_score": None,
        "error_message": "SyntaxError: Missing required function 'inner_steps'",
    },
    "sub_fail_eval": {
        "submission_id": "sub_fail_eval",
        "miner_hotkey": "5FailEvalHotkey1234567890abcdef1234567890abcd",
        "miner_uid": 55,
        "code_hash": "evalfail123456789012345678901234567890abc",
        "status": "failed_evaluation",
        "created_at": "2026-01-07T08:40:00Z",
        "updated_at": "2026-01-07T08:42:00Z",
        "final_score": None,
        "error_message": "LogitsMismatchError: Output vectors differ by 15% (threshold: 2%)",
    },
    "sub_timeout": {
        "submission_id": "sub_timeout",
        "miner_hotkey": "5TimeoutHotkey123456789012345678901234567890abcd",
        "miner_uid": 77,
        "code_hash": "timeout12345678901234567890123456789012345",
        "status": "failed_evaluation",
        "created_at": "2026-01-07T08:35:00Z",
        "updated_at": "2026-01-07T08:35:01Z",
        "final_score": None,
        "error_message": "Evaluation timed out after 600s",
    },
}

# Evaluations keyed by submission_id
MOCK_EVALUATIONS = {
    "sub_001": [
        {
            "evaluation_id": "eval_001",
            "submission_id": "sub_001",
            "evaluator_hotkey": "5Validator...",
            "mfu": 43.2,  # MFU %
            "tokens_per_second": 4315,
            "total_tokens": 40960,
            "wall_time_seconds": 9.49,
            "success": True,
            "error": None,
            "created_at": "2026-01-07T10:32:00Z",
        },
        {
            "evaluation_id": "eval_002",
            "submission_id": "sub_001",
            "evaluator_hotkey": "5Validator...",
            "mfu": 42.9,  # MFU %
            "tokens_per_second": 4289,
            "total_tokens": 40960,
            "wall_time_seconds": 9.55,
            "success": True,
            "error": None,
            "created_at": "2026-01-07T10:33:00Z",
        },
        {
            "evaluation_id": "eval_003",
            "submission_id": "sub_001",
            "evaluator_hotkey": "5Validator...",
            "mfu": 43.0,  # MFU %
            "tokens_per_second": 4302,
            "total_tokens": 40960,
            "wall_time_seconds": 9.52,
            "success": True,
            "error": None,
            "created_at": "2026-01-07T10:34:00Z",
        },
    ],
    "sub_002": [
        {
            "evaluation_id": "eval_004",
            "submission_id": "sub_002",
            "evaluator_hotkey": "5Validator...",
            "mfu": 41.6,  # MFU %
            "tokens_per_second": 4160,
            "total_tokens": 40960,
            "wall_time_seconds": 9.85,
            "success": True,
            "error": None,
            "created_at": "2026-01-07T09:17:00Z",
        },
        {
            "evaluation_id": "eval_005",
            "submission_id": "sub_002",
            "evaluator_hotkey": "5Validator...",
            "mfu": 41.5,  # MFU %
            "tokens_per_second": 4152,
            "total_tokens": 40960,
            "wall_time_seconds": 9.87,
            "success": True,
            "error": None,
            "created_at": "2026-01-07T09:18:00Z",
        },
        {
            "evaluation_id": "eval_006",
            "submission_id": "sub_002",
            "evaluator_hotkey": "5Validator...",
            "mfu": 41.6,  # MFU %
            "tokens_per_second": 4156,
            "total_tokens": 40960,
            "wall_time_seconds": 9.86,
            "success": True,
            "error": None,
            "created_at": "2026-01-07T09:19:00Z",
        },
    ],
}

MOCK_CODE = '''"""
Optimized training code for Templar Crusades.
Achieves 43% MFU through efficient memory management and kernel fusion.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class InnerStepsResult:
    """Required return type from inner_steps function."""

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


def inner_steps(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
) -> InnerStepsResult:
    """
    Run training for num_steps and return results.

    Optimizations applied:
    - torch.compile for kernel fusion
    - Mixed precision with bfloat16
    - Gradient accumulation batching
    - CUDA graph capture for repeated operations
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    # Pre-allocate buffers for efficiency
    model = torch.compile(model, mode="reduce-overhead")

    for step in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long, non_blocking=True)

        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Efficient cross-entropy with fused kernel
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )

        # Gradient computation
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = loss.item()

    # Ensure all CUDA operations complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
'''


def get_default_submission(submission_id: str) -> dict:
    """Get a default submission for unknown IDs."""
    return {
        "submission_id": submission_id,
        "miner_hotkey": "5Unknown...",
        "miner_uid": 0,
        "code_hash": "unknown",
        "status": "finished",
        "created_at": "2026-01-07T10:00:00Z",
        "updated_at": "2026-01-07T10:05:00Z",
        "final_score": 35.0,  # MFU %
        "error_message": None,
    }


def get_default_evaluations(submission_id: str) -> list:
    """Get default evaluations for unknown IDs."""
    return [
        {
            "evaluation_id": f"eval_{submission_id}_1",
            "submission_id": submission_id,
            "evaluator_hotkey": "5Validator...",
            "mfu": 35.0,  # MFU %
            "tokens_per_second": 3500,
            "total_tokens": 40960,
            "wall_time_seconds": 11.7,
            "success": True,
            "error": None,
            "created_at": "2026-01-07T10:02:00Z",
        },
        {
            "evaluation_id": f"eval_{submission_id}_2",
            "submission_id": submission_id,
            "evaluator_hotkey": "5Validator...",
            "mfu": 35.0,  # MFU %
            "tokens_per_second": 3495,
            "total_tokens": 40960,
            "wall_time_seconds": 11.72,
            "success": True,
            "error": None,
            "created_at": "2026-01-07T10:03:00Z",
        },
        {
            "evaluation_id": f"eval_{submission_id}_3",
            "submission_id": submission_id,
            "evaluator_hotkey": "5Validator...",
            "mfu": 35.1,  # MFU %
            "tokens_per_second": 3505,
            "total_tokens": 40960,
            "wall_time_seconds": 11.68,
            "success": True,
            "error": None,
            "created_at": "2026-01-07T10:04:00Z",
        },
    ]
