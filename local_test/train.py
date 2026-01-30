"""
Basic training implementation - Miners can optimize this!

Usage:
    1. Run setup: uv run local_test/setup_benchmark.py
    2. Test locally: uv run local_test/train.py
    3. Submit when ready!
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@dataclass
class InnerStepsResult:
    """Required return type from inner_steps function."""

    final_logits: torch.Tensor  # Output logits from last forward pass
    total_tokens: int  # Total tokens processed across all steps
    final_loss: float  # Loss value from last training step


# Reuse compiled model across warmup and timed run (same process, same model id).
_compiled_model_cache: dict[int, torch.nn.Module] = {}


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """
    Run training steps and return results.

    Args:
        model: Pre-loaded model (already on device, in train mode)
        data_iterator: Iterator yielding batches of shape (batch_size, seq_len)
        optimizer: Pre-configured optimizer
        num_steps: Number of training steps to run
        device: Target device (cuda or cpu)

    Returns:
        InnerStepsResult with outputs for verification
    """
    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    # Reuse compiled model so warmup pays compile cost; timed run reuses cache.
    model_id = id(model)
    if model_id not in _compiled_model_cache:
        _compiled_model_cache[model_id] = torch.compile(model, mode="reduce-overhead")
    model = _compiled_model_cache[model_id]

    ce_loss = F.cross_entropy

    # Separate CUDA stream for prefetch (overlaps next-batch transfer with backward)
    prefetch_stream = torch.cuda.Stream() if device.type == "cuda" else None

    batch = next(data_iterator)
    if batch.device != device:
        batch = batch.to(device, dtype=torch.long, non_blocking=True)

    next_batch = None
    _get_logits = None

    for step in range(num_steps):
        if prefetch_stream is not None and next_batch is not None:
            prefetch_stream.synchronize()
        batch = next_batch if next_batch is not None else batch

        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        if step < num_steps - 1:
            if prefetch_stream is not None:
                with torch.cuda.stream(prefetch_stream):
                    next_batch = next(data_iterator)
                    if next_batch.device != device:
                        next_batch = next_batch.to(device, dtype=torch.long, non_blocking=True)
            else:
                next_batch = next(data_iterator)
                if next_batch.device != device:
                    next_batch = next_batch.to(device, dtype=torch.long, non_blocking=True)
        else:
            next_batch = None

        outputs = model(input_ids, use_cache=False)
        if _get_logits is None:
            _get_logits = (lambda o: o.logits) if hasattr(outputs, "logits") else (lambda o: o)
        logits = _get_logits(outputs)

        vocab_size = logits.size(-1)
        try:
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
        except RuntimeError:
            logits_flat = logits.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)

        loss = ce_loss(logits_flat, labels_flat, ignore_index=-100)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()

        if step == num_steps - 1:
            final_logits = logits.detach().float()
            final_loss = loss.item()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )


# =============================================================================
# LOCAL TESTING - Run this file to test your implementation
# =============================================================================
if __name__ == "__main__":
    # Speed opts for local runs only (validators use their own settings)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print("TESTING train.py - Basic Implementation")
    print("=" * 60)
    print()

    # Load configuration
    hparams_path = Path(__file__).parent.parent / "hparams" / "hparams.json"
    hparams = {}
    if hparams_path.exists():
        with open(hparams_path) as f:
            hparams = json.load(f)

    batch_size = hparams.get("benchmark_batch_size", 16)
    num_steps = hparams.get("eval_steps", 5)
    num_evals = hparams.get("evaluation_runs", 5)

    print(f"Batch size: {batch_size}")
    print(f"Steps per eval: {num_steps}")
    print(f"Evaluations: {num_evals}")
    print()

    # Check paths
    project_root = Path(__file__).parent.parent
    model_path = project_root / "benchmark" / "model"
    data_path = project_root / "benchmark" / "data" / "train.pt"

    if not model_path.exists() or not data_path.exists():
        print("Setup required! Run: uv run local_test/setup_benchmark.py")
        exit(1)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()  # Required to fit in GPU memory
    model.train()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Load data; pin_memory speeds up CPU->GPU transfer when using non_blocking=True
    print("Loading data...")
    data = torch.load(data_path, weights_only=True)
    if torch.cuda.is_available():
        data = data.pin_memory()
    print(f"Samples: {data.shape[0]:,}, Sequence length: {data.shape[1]}")
    print()

    # Create data iterator
    def create_iterator():
        idx = 0
        while True:
            end_idx = idx + batch_size
            if end_idx > data.shape[0]:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx

    # Fused AdamW can be faster on CUDA (PyTorch 2.0+)
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, fused=torch.cuda.is_available()
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup (compile happens here; timed run reuses compiled model)
    print("Warmup...")
    _ = inner_steps(model, create_iterator(), optimizer, num_steps=2, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print()

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, fused=torch.cuda.is_available()
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Run evaluations
    print(f"Running {num_evals} evaluations...")

    for i in range(num_evals):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = inner_steps(model, create_iterator(), optimizer, num_steps, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        print(
            f"  Eval {i + 1}: {elapsed:.3f}s, tokens={result.total_tokens:,}, loss={result.final_loss:.4f}"
        )

    print()
    print("Done!")