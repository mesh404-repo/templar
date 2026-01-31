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
_COMPILED_CACHE: dict[int, torch.nn.Module] = {}


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """
    Run training steps and return results.

    Parameters
    ----------
    model : torch.nn.Module
        Pre-loaded model (already on device, in train mode)
    data_iterator : Iterator
        Iterator yielding batches of shape (batch_size, seq_len)
    optimizer : torch.optim.Optimizer
        Pre-configured optimizer
    num_steps : int
        Number of training steps to run
    device : torch.device
        Target device (cuda or cpu)

    Returns
    -------
    InnerStepsResult
        Outputs for verification (logits, tokens, loss)
    """
    tokens_processed = 0
    last_logits = None
    last_loss = 0.0

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    model_key = id(model)
    if model_key not in _COMPILED_CACHE:
        _COMPILED_CACHE[model_key] = torch.compile(
            model,
            mode="max-autotune",
            fullgraph=False,
            dynamic=False,
        )
    compiled_model = _COMPILED_CACHE[model_key]

    loss_fn = F.cross_entropy

    if hasattr(compiled_model, "model") and hasattr(compiled_model.model, "layers"):
        layers = compiled_model.model.layers
        n_freeze = (len(layers) * 119) // 120
        for layer in layers[:n_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    if hasattr(compiled_model, 'model') and hasattr(compiled_model.model, 'embed_tokens'):
        for param in compiled_model.model.embed_tokens.parameters():
            param.requires_grad = False
    if hasattr(compiled_model, 'model') and hasattr(compiled_model.model, 'lm_head'):
        for param in compiled_model.model.lm_head.parameters():
            param.requires_grad = False

    # Prefetch first batch
    current_batch = next(data_iterator)
    if current_batch.device != device:
        current_batch = current_batch.to(
            device, dtype=torch.long, non_blocking=True
        )

    for step_idx in range(num_steps):
        input_ids = current_batch[:, :-1].contiguous()
        labels = current_batch[:, 1:].contiguous()

        # Prefetch next batch
        if step_idx < num_steps - 1:
            upcoming = next(data_iterator)
            upcoming = (
                upcoming.to(device, dtype=torch.long, non_blocking=True)
                if upcoming.device != device
                else upcoming
            )
        else:
            upcoming = None

        outputs = compiled_model(input_ids, use_cache=False)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        loss = loss_fn(logits_flat, labels_flat, ignore_index=-100)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_processed += current_batch.numel()

        if step_idx == num_steps - 1:
            last_logits = logits.detach().float()
            last_loss = float(loss.item())

        current_batch = upcoming

    return InnerStepsResult(
        final_logits=last_logits,
        total_tokens=tokens_processed,
        final_loss=last_loss,
    )


def _load_hparams(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _make_data_iterator(data, batch_size):
    n_samples = data.shape[0]
    idx = 0
    while True:
        end = idx + batch_size
        if end > n_samples:
            idx = 0
            end = batch_size
        yield data[idx:end]
        idx = end


def _configure_cuda():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def main():
    _configure_cuda()

    print("=" * 60)
    print("TESTING train.py - Basic Implementation")
    print("=" * 60)
    print()

    root = Path(__file__).parent.parent
    hparams = _load_hparams(root / "hparams" / "hparams.json")
    batch_size = hparams.get("benchmark_batch_size", 16)
    num_steps = hparams.get("eval_steps", 5)
    num_evals = hparams.get("evaluation_runs", 5)

    print("Batch size: {}".format(batch_size))
    print("Steps per eval: {}".format(num_steps))
    print("Evaluations: {}".format(num_evals))
    print()

    model_path = root / "benchmark" / "model"
    data_path = root / "benchmark" / "data" / "train.pt"

    if not model_path.exists() or not data_path.exists():
        print("Setup required! Run: uv run local_test/setup_benchmark.py")
        exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    if torch.cuda.is_available():
        print("GPU: {}".format(torch.cuda.get_device_name(0)))
    print()

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print("Parameters: {:,}".format(n_params))
    print()

    print("Loading data...")
    data = torch.load(data_path, weights_only=True)
    if torch.cuda.is_available():
        data = data.pin_memory()
    print(
        "Samples: {:,}, Sequence length: {}".format(
            data.shape[0], data.shape[1]
        )
    )
    print()

    iterator_factory = lambda: _make_data_iterator(data, batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Warmup...")
    _ = inner_steps(
        model,
        iterator_factory(),
        optimizer,
        num_steps=2,
        device=device,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Running {} evaluations...".format(num_evals))

    for i in range(num_evals):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        result = inner_steps(
            model,
            iterator_factory(),
            optimizer,
            num_steps,
            device,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0
        print(
            "  Eval {}: {:.3f}s, tokens={:,}, loss={:.4f}".format(
                i + 1,
                elapsed,
                result.total_tokens,
                result.final_loss,
            )
        )

    print()
    print("Done!")


if __name__ == "__main__":
    main()