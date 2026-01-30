import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


# =========================
# GLOBAL CUDA OPTS (SAFE)
# =========================
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    # Local refs (avoid attribute lookup cost)
    ce_loss = F.cross_entropy

    # Compile model forward pass for speed (cache compiled function per model)
    # Use model id as cache key to avoid recompiling same model
    if not hasattr(inner_steps, "_compiled_forwards"):
        inner_steps._compiled_forwards = {}
    
    model_id = id(model)
    if model_id not in inner_steps._compiled_forwards:
        # Compile forward pass (only forward, not backward, for training compatibility)
        def _forward(input_ids):
            return model(input_ids)
        
        # Compile with mode="reduce-overhead" for training loops (faster subsequent calls)
        if device.type == "cuda" and hasattr(torch, "compile"):
            try:
                inner_steps._compiled_forwards[model_id] = torch.compile(_forward, mode="reduce-overhead")
            except Exception:
                # Fallback if compile fails
                inner_steps._compiled_forwards[model_id] = _forward
        else:
            inner_steps._compiled_forwards[model_id] = _forward
    
    compiled_forward = inner_steps._compiled_forwards[model_id]

    # Use CUDA stream for prefetching if available
    prefetch_stream = None
    if device.type == "cuda":
        prefetch_stream = torch.cuda.Stream()

    # Prefetch first batch
    batch = next(data_iterator)
    if batch.device != device:
        batch = batch.to(device, non_blocking=True)

    # Cache logits access pattern on first forward pass
    _has_logits_attr = None
    _get_logits = None

    next_batch = None
    for step in range(num_steps):
        # Sync prefetch stream before using prefetched batch (if any)
        if prefetch_stream is not None and next_batch is not None:
            prefetch_stream.synchronize()
        
        batch = next_batch if next_batch is not None else batch
        
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        # Start prefetching next batch in separate stream (overlaps with backward pass)
        if step < num_steps - 1:
            if prefetch_stream is not None:
                with torch.cuda.stream(prefetch_stream):
                    next_batch = next(data_iterator)
                    if next_batch.device != device:
                        next_batch = next_batch.to(device, non_blocking=True)
            else:
                next_batch = next(data_iterator)
                if next_batch.device != device:
                    next_batch = next_batch.to(device, non_blocking=True)
        else:
            next_batch = None

        # Forward pass using compiled function (faster execution)
        outputs = compiled_forward(input_ids)
        
        # Cache logits access pattern on first iteration
        if _has_logits_attr is None:
            _has_logits_attr = hasattr(outputs, "logits")
            _get_logits = (lambda o: o.logits) if _has_logits_attr else (lambda o: o)
        
        logits = _get_logits(outputs)

        # Use view() instead of reshape() for contiguous tensors (faster, zero-copy)
        # Logits from transformer models are typically contiguous
        # Fallback to reshape if view fails (shouldn't happen for transformer logits)
        try:
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
        except RuntimeError:
            # Fallback if tensors are not contiguous (unlikely for transformer outputs)
            logits_flat = logits.reshape(-1, logits.size(-1))
            labels_flat = labels.reshape(-1)

        loss = ce_loss(logits_flat, labels_flat, ignore_index=-100)

        # Backward pass (prefetch can overlap with this)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()

        if step == num_steps - 1:
            final_logits = logits.detach()
            final_loss = loss.item()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )


# =========================
# LOCAL TEST
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING train.py (IMPROVED TPS)")
    print("=" * 60)
    print()

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

    project_root = Path(__file__).parent.parent
    model_path = project_root / "benchmark" / "model"
    data_path = project_root / "benchmark" / "data" / "train.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.train()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    print("Loading data...")
    data = torch.load(data_path, weights_only=True)
    # Pin memory for faster transfers
    if torch.cuda.is_available():
        data = data.pin_memory()
    print(f"Samples: {data.shape[0]:,}, Seq len: {data.shape[1]}")
    print()

    def create_iterator():
        idx = 0
        while True:
            end = idx + batch_size
            if end > data.shape[0]:
                idx = 0
                end = batch_size
            yield data[idx:end]
            idx = end

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Warmup...")
    _ = inner_steps(model, create_iterator(), optimizer, num_steps=2, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    total_time = 0.0
    total_tokens = 0
    tokens_per_second_list = []

    print(f"Running {num_evals} evaluations...\n")

    for i in range(num_evals):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = inner_steps(model, create_iterator(), optimizer, num_steps, device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        tps = result.total_tokens / elapsed

        total_time += elapsed
        total_tokens += result.total_tokens
        tokens_per_second_list.append(tps)

        print(
            f"Eval {i+1}: {elapsed:.3f}s | "
            f"tokens={result.total_tokens:,} | "
            f"TPS={tps:,.0f} | "
            f"loss={result.final_loss:.4f}"
        )

    print()
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.3f}s")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens/second: {total_tokens / total_time:,.0f}")

    if tokens_per_second_list:
        print(f"Min tokens/s: {min(tokens_per_second_list):,.0f}")
        print(f"Max tokens/s: {max(tokens_per_second_list):,.0f}")
        print(f"Avg tokens/s: {sum(tokens_per_second_list)/len(tokens_per_second_list):,.0f}")

    print(f"Average time per eval: {total_time / num_evals:.3f}s")
    print()