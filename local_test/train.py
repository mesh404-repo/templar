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

_compiled_model_cache = {}

def inner_steps(model, data_iterator, optimizer, num_steps, device):
    global _compiled_model_cache

    model_id = id(model)
    if model_id not in _compiled_model_cache:
        _compiled_model_cache[model_id] = torch.compile(model, mode="reduce-overhead")

    model = _compiled_model_cache[model_id]

    total_tokens = 0
    final_logits = None
    final_loss = 0.0

    # Local refs (avoid attribute lookup cost)
    ce_loss = F.cross_entropy
    model_train = model.train
    tensor_numel = torch.Tensor.numel
    tensor_reshape = torch.Tensor.reshape

    # Prefetch first batch
    batch = next(data_iterator)
    if batch.device != device:
        batch = batch.to(device, non_blocking=True)

    for step in range(num_steps):
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        # Prefetch next batch while computing current
        if step < num_steps - 1:
            next_batch = next(data_iterator)
            if next_batch.device != device:
                next_batch = next_batch.to(device, non_blocking=True)
        else:
            next_batch = None

        # Forward pass (model already in bfloat16)
        outputs = model(input_ids)
        logits = outputs.logits

        loss = ce_loss(
            tensor_reshape(logits, (-1, logits.size(-1))),
            tensor_reshape(labels, (-1,)),
            ignore_index=-100,
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += tensor_numel(batch)

        if step == num_steps - 1:
            final_logits = logits.detach()
            final_loss = loss.item()

        batch = next_batch

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True if torch.cuda.is_available() else False)

    print("Warmup...")
    _ = inner_steps(model, create_iterator(), optimizer, num_steps=2, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True if torch.cuda.is_available() else False)

    total_time = 0.0
    total_tokens = 0
    tokens_per_second_list = []

    print(f"Running {num_evals} evaluations...\n")

    for i in range(num_evals):
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