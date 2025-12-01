"""
Reference inner_steps implementation for templar-tournament.

This is the canonical implementation that miners must match in terms of
output logits, token counts, and loss values. Miners can optimize this
implementation but must produce the same outputs.

Usage:
    from benchmark.reference_inner_steps import inner_steps, InnerStepsResult

    result = inner_steps(model, data_iterator, optimizer, num_steps, device)
"""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class InnerStepsResult:
    """Result from inner_steps function.

    Miners must return this from their inner_steps function.

    Attributes:
        final_logits: Output logits from the last forward pass.
                      Shape: (batch_size, seq_len, vocab_size)
        total_tokens: Total number of tokens processed across all steps.
        final_loss: Loss value from the last training step.
    """

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


def inner_steps(
    model: nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
) -> InnerStepsResult:
    """Execute N training steps and return results for verification.

    This is the reference implementation. Your optimized version must
    produce the same final_logits, total_tokens, and final_loss values.

    Args:
        model: Pre-loaded model (already on device, in train mode).
        data_iterator: Iterator yielding input tensors of shape (batch_size, seq_len).
        optimizer: Pre-configured optimizer (AdamW with standard hyperparams).
        num_steps: Number of training steps to run.
        device: Target device (cuda/cpu).

    Returns:
        InnerStepsResult containing:
        - final_logits: Output logits from last forward pass
        - total_tokens: Total tokens processed
        - final_loss: Loss value from last step

    Example:
        >>> model = load_model()
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> data_iter = create_data_iterator(data_path, batch_size=8, seq_len=1024)
        >>> result = inner_steps(model, data_iter, optimizer, num_steps=100, device=device)
        >>> print(f"TPS: {result.total_tokens / elapsed_time}")
    """
    # Ensure deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        # Get next batch
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        # Prepare inputs and labels for causal LM
        # Input: tokens[:-1], Labels: tokens[1:] (next-token prediction)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        # Forward pass with bf16 autocast
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(input_ids)

            # Cross-entropy loss for next-token prediction
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Track metrics
        total_tokens += batch.numel()
        final_logits = logits.detach().float()  # Convert to fp32 for comparison
        final_loss = loss.item()

    # Sync GPU before returning
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )


# ============================================================================
# OPTIMIZATION GUIDELINES
# ============================================================================
#
# You can optimize this implementation in many ways, as long as you produce
# the same outputs (within tolerance):
#
# 1. FORWARD PASS OPTIMIZATIONS:
#    - Use torch.compile() for kernel fusion
#    - Implement custom CUDA kernels
#    - Use flash attention if model supports it
#    - Apply gradient checkpointing
#
# 2. DATA LOADING OPTIMIZATIONS:
#    - Use pinned memory for faster H2D transfers
#    - Prefetch next batch while current is processing
#    - Use non-blocking transfers
#
# 3. OPTIMIZER OPTIMIZATIONS:
#    - Use fused optimizers (FusedAdam, FusedLamb)
#    - Combine backward and optimizer step
#
# 4. MEMORY OPTIMIZATIONS:
#    - Use gradient accumulation efficiently
#    - Implement activation checkpointing
#    - Use memory-efficient attention
#
# IMPORTANT: Your optimizations must NOT change:
#    - The model architecture
#    - The loss function (cross_entropy)
#    - The input/output format
#    - The label preparation (next-token prediction)
#
# Your outputs are compared against the reference with these tolerances:
#    - Logits: atol=1e-3, rtol=1e-3
#    - Token count: exact match
#    - Loss: tolerance=1e-3
#
# ============================================================================
