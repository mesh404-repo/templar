"""
Verify your train.py passes validator checks before submitting.

Usage:
    uv run local_test/verify.py

This script runs the SAME checks as the validator:
1. Token count matches expected
2. Loss is valid and similar to reference
3. 100% of parameters are trainable (no frozen layers)
4. 80% of parameters change during training
5. Gradient verification (cosine similarity, norm ratio)

Fix any failures before submitting to avoid failed evaluations!
"""

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class InnerStepsResult:
    """Result type for verification (mirrors train.py's InnerStepsResult)."""

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


@dataclass
class GradientInfo:
    """Gradient information for verification."""

    norm: float
    vector: torch.Tensor | None
    layers_with_grad: int
    total_layers: int


def load_train_module(train_path: Path):
    """Load train.py as a module."""
    spec = importlib.util.spec_from_file_location("train", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_gradients(model: torch.nn.Module) -> GradientInfo:
    """Capture gradient information from model after backward pass."""
    grads = []
    layers_with_grad = 0
    total_layers = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_layers += 1
            if param.grad is not None:
                grad = param.grad.detach().float().flatten()
                if grad.abs().sum() > 0:
                    layers_with_grad += 1
                grads.append(grad)

    if grads:
        grad_vector = torch.cat(grads)
        grad_norm = grad_vector.norm().item()
    else:
        grad_vector = None
        grad_norm = 0.0

    return GradientInfo(
        norm=grad_norm,
        vector=grad_vector,
        layers_with_grad=layers_with_grad,
        total_layers=total_layers,
    )


def run_reference_with_gradients(model, data_iterator, optimizer, num_steps, device):
    """Run reference training and capture gradients on final step."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    total_tokens = 0
    final_logits = None
    final_loss = 0.0
    grad_info = None

    for step in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        loss.backward()

        # Capture gradients on final step BEFORE optimizer.step()
        if step == num_steps - 1:
            grad_info = capture_gradients(model)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = float(loss.item())

    result = InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
    return result, grad_info


def verify_trainable_params(model, min_ratio: float = 1.0) -> tuple[bool, str | None, dict]:
    """Check that required percentage of parameters are trainable."""
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    ratio = trainable_params / total_params if total_params > 0 else 0.0
    details = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": ratio,
        "required_ratio": min_ratio,
    }

    if ratio < min_ratio:
        error = f"Only {ratio:.1%} params trainable, need {min_ratio:.0%}"
        return False, error, details

    return True, None, details


def verify_params_changed(
    model, initial_state: dict, min_ratio: float = 0.8
) -> tuple[bool, str | None, dict]:
    """Check that required percentage of parameters changed during training."""
    total_params = 0
    changed_params = 0

    for name, param in model.named_parameters():
        if name in initial_state:
            total_params += param.numel()
            diff = (param.data - initial_state[name].to(param.device)).abs()
            changed_params += (diff > 1e-8).sum().item()

    ratio = changed_params / total_params if total_params > 0 else 0.0
    details = {
        "total_params": total_params,
        "changed_params": changed_params,
        "changed_ratio": ratio,
        "required_ratio": min_ratio,
    }

    if ratio < min_ratio:
        error = f"Only {ratio:.1%} params changed, need {min_ratio:.0%}"
        return False, error, details

    return True, None, details


def verify_gradients(
    reference_grad: GradientInfo,
    candidate_grad: GradientInfo,
    cosine_min: float = 0.8,
    norm_ratio_min: float = 0.5,
    norm_ratio_max: float = 2.0,
) -> tuple[bool, str | None, dict]:
    """Verify gradient similarity between reference and candidate."""
    details = {
        "reference_norm": reference_grad.norm,
        "candidate_norm": candidate_grad.norm,
        "checks_passed": [],
        "checks_failed": [],
    }

    # Check gradient coverage (100% of layers must have gradients)
    if candidate_grad.total_layers > 0:
        coverage = candidate_grad.layers_with_grad / candidate_grad.total_layers
        details["gradient_coverage"] = coverage
        details["layers_with_grad"] = candidate_grad.layers_with_grad
        details["total_layers"] = candidate_grad.total_layers

        if coverage < 1.0:
            error = (
                f"Not all layers have gradients: "
                f"{candidate_grad.layers_with_grad}/{candidate_grad.total_layers}"
            )
            details["checks_failed"].append({"check": "gradient_coverage", "error": error})
            return False, error, details
        details["checks_passed"].append("gradient_coverage")

    # Check gradient norm ratio
    if reference_grad.norm > 0:
        norm_ratio = candidate_grad.norm / reference_grad.norm
        details["norm_ratio"] = norm_ratio

        if norm_ratio < norm_ratio_min or norm_ratio > norm_ratio_max:
            error = (
                f"Gradient norm ratio {norm_ratio:.2f} outside [{norm_ratio_min}, {norm_ratio_max}]"
            )
            details["checks_failed"].append({"check": "norm_ratio", "error": error})
            return False, error, details
        details["checks_passed"].append("norm_ratio")

    # Check cosine similarity
    if reference_grad.vector is not None and candidate_grad.vector is not None:
        if reference_grad.vector.shape == candidate_grad.vector.shape:
            cosine = F.cosine_similarity(
                reference_grad.vector.unsqueeze(0),
                candidate_grad.vector.unsqueeze(0),
            ).item()
            details["cosine_similarity"] = cosine

            if cosine < cosine_min:
                error = f"Gradient cosine similarity {cosine:.4f} < {cosine_min}"
                details["checks_failed"].append({"check": "cosine_similarity", "error": error})
                return False, error, details
            details["checks_passed"].append("cosine_similarity")

    return True, None, details


def verify_all(
    reference,
    candidate,
    expected_tokens: int,
    initial_state: dict,
    model,
    reference_grad: GradientInfo,
    candidate_grad: GradientInfo,
    max_loss_difference: float = 0.5,
    min_trainable_ratio: float = 1.0,
    min_changed_ratio: float = 0.8,
    gradient_cosine_min: float = 0.8,
    gradient_norm_ratio_min: float = 0.5,
    gradient_norm_ratio_max: float = 2.0,
) -> bool:
    """Run all validator checks - SAME as production validator."""
    print()
    print("=" * 70)
    print("VERIFICATION: Running validator checks (same as production)")
    print("=" * 70)

    all_passed = True

    # CHECK 1: Token count
    print("\n[CHECK 1/5] Token count")
    print(f"  Expected: {expected_tokens}, Got: {candidate.total_tokens}")
    if candidate.total_tokens != expected_tokens:
        print("  [FAILED] Token count mismatch!")
        all_passed = False
    else:
        print("  [PASSED]")

    # CHECK 2: Loss validity and comparison
    print("\n[CHECK 2/5] Loss validity and comparison")
    print(f"  Reference loss: {reference.final_loss:.6f}")
    print(f"  Candidate loss: {candidate.final_loss:.6f}")

    if candidate.final_loss != candidate.final_loss:  # NaN check
        print("  [FAILED] Loss is NaN!")
        all_passed = False
    elif abs(candidate.final_loss) > 100:
        print(f"  [FAILED] Loss unreasonable: {candidate.final_loss:.4f}")
        all_passed = False
    else:
        loss_diff = abs(candidate.final_loss - reference.final_loss)
        print(f"  Loss difference: {loss_diff:.4f} (max allowed: {max_loss_difference})")
        if loss_diff > max_loss_difference:
            print("  [FAILED] Loss difference too large!")
            all_passed = False
        else:
            print("  [PASSED]")

    # CHECK 3: Trainable parameters (100% required)
    print(f"\n[CHECK 3/5] Trainable parameters (need {min_trainable_ratio:.0%})")
    trainable_ok, trainable_err, trainable_details = verify_trainable_params(
        model, min_trainable_ratio
    )
    trainable_count = trainable_details["trainable_params"]
    total_count = trainable_details["total_params"]
    print(f"  Trainable: {trainable_count:,} / {total_count:,}")
    print(f"  Ratio: {trainable_details['trainable_ratio']:.1%}")
    if not trainable_ok:
        print(f"  [FAILED] {trainable_err}")
        all_passed = False
    else:
        print("  [PASSED]")

    # CHECK 4: Parameters changed (80% required)
    print(f"\n[CHECK 4/5] Parameters changed (need {min_changed_ratio:.0%})")
    changed_ok, changed_err, changed_details = verify_params_changed(
        model, initial_state, min_changed_ratio
    )
    changed_count = changed_details["changed_params"]
    total_params = changed_details["total_params"]
    print(f"  Changed: {changed_count:,} / {total_params:,}")
    print(f"  Ratio: {changed_details['changed_ratio']:.1%}")
    if not changed_ok:
        print(f"  [FAILED] {changed_err}")
        all_passed = False
    else:
        print("  [PASSED]")

    # CHECK 5: Gradient verification
    print("\n[CHECK 5/5] Gradient verification")
    grad_ok, grad_err, grad_details = verify_gradients(
        reference_grad,
        candidate_grad,
        cosine_min=gradient_cosine_min,
        norm_ratio_min=gradient_norm_ratio_min,
        norm_ratio_max=gradient_norm_ratio_max,
    )
    print(f"  Reference gradient norm: {grad_details['reference_norm']:.4f}")
    print(f"  Candidate gradient norm: {grad_details['candidate_norm']:.4f}")
    if "gradient_coverage" in grad_details:
        cov = grad_details["gradient_coverage"]
        layers = grad_details["layers_with_grad"]
        total = grad_details["total_layers"]
        print(f"  Gradient coverage: {cov:.1%} ({layers}/{total} layers)")
    if "norm_ratio" in grad_details:
        ratio = grad_details["norm_ratio"]
        allowed = f"{gradient_norm_ratio_min}-{gradient_norm_ratio_max}"
        print(f"  Norm ratio: {ratio:.4f} (allowed: {allowed})")
    if "cosine_similarity" in grad_details:
        cos = grad_details["cosine_similarity"]
        print(f"  Cosine similarity: {cos:.4f} (min: {gradient_cosine_min})")

    if not grad_ok:
        print(f"  [FAILED] {grad_err}")
        all_passed = False
    else:
        print("  [PASSED]")

    # Summary
    print()
    print("=" * 70)
    if all_passed:
        print("VERIFICATION: ALL CHECKS PASSED")
        print("Your submission should pass validator evaluation!")
    else:
        print("VERIFICATION: SOME CHECKS FAILED")
        print("Fix the issues above before submitting.")
    print("=" * 70)

    return all_passed


def main():
    print("=" * 70)
    print("VERIFYING train.py - Same checks as production validator")
    print("=" * 70)
    print()

    # Load configuration
    project_root = Path(__file__).parent.parent
    hparams_path = project_root / "hparams" / "hparams.json"
    hparams = {}
    if hparams_path.exists():
        with open(hparams_path) as f:
            hparams = json.load(f)

    batch_size = hparams.get("benchmark_batch_size", 4)
    seq_len = hparams.get("benchmark_sequence_length", 1024)
    num_steps = hparams.get("eval_steps", 5)

    # Verification settings (same as validator)
    verification = hparams.get("verification", {})
    max_loss_difference = verification.get("max_loss_difference", 0.5)
    min_trainable_ratio = 1.0
    min_changed_ratio = verification.get("min_params_changed_ratio", 0.8)
    gradient_cosine_min = verification.get("gradient_cosine_min", 0.8)
    gradient_norm_ratio_min = verification.get("gradient_norm_ratio_min", 0.5)
    gradient_norm_ratio_max = verification.get("gradient_norm_ratio_max", 2.0)

    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Steps per eval: {num_steps}")
    print(f"  Max loss difference: {max_loss_difference}")
    print(f"  Min trainable params: {min_trainable_ratio:.0%}")
    print(f"  Min params changed: {min_changed_ratio:.0%}")
    print(f"  Gradient cosine min: {gradient_cosine_min}")
    print(f"  Gradient norm ratio: {gradient_norm_ratio_min}-{gradient_norm_ratio_max}")
    print()

    # Check paths
    model_path = project_root / "benchmark" / "model"
    data_path = project_root / "benchmark" / "data" / "train.pt"
    train_path = project_root / "local_test" / "train.py"

    if not model_path.exists() or not data_path.exists():
        print("Setup required! Run: uv run local_test/setup_benchmark.py")
        sys.exit(1)

    if not train_path.exists():
        print(f"train.py not found at {train_path}")
        sys.exit(1)

    # Load miner's train.py module
    print("Loading train.py...")
    train_module = load_train_module(train_path)

    if not hasattr(train_module, "inner_steps"):
        print("ERROR: train.py must have an 'inner_steps' function!")
        sys.exit(1)

    if not hasattr(train_module, "InnerStepsResult"):
        print("ERROR: train.py must have an 'InnerStepsResult' dataclass!")
        sys.exit(1)

    print("  Found inner_steps function")
    print("  Found InnerStepsResult dataclass")
    print()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model with RANDOM INIT (same as validator - anti-cheat measure)
    print("Loading model with RANDOM INITIALIZATION (same as validator)...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.gradient_checkpointing_enable()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("NOTE: Using random weights, not pretrained (anti-cheat)")
    print()

    # Load data
    print("Loading data...")
    data = torch.load(data_path, weights_only=True)
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

    # Save initial model state (BEFORE any training)
    print("Saving initial model state...")
    initial_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    print()

    # Expected tokens = batch_size * seq_len * num_steps
    expected_tokens = batch_size * seq_len * num_steps

    # Run reference with gradient capture (same as validator)
    print("Running reference baseline (with gradient capture)...")
    model.train()
    optimizer_ref = torch.optim.AdamW(model.parameters(), lr=1e-4)
    reference, reference_grad = run_reference_with_gradients(
        model, create_iterator(), optimizer_ref, num_steps, device
    )
    print(f"  Reference loss: {reference.final_loss:.6f}")
    print(f"  Reference tokens: {reference.total_tokens:,}")
    print(f"  Reference gradient norm: {reference_grad.norm:.4f}")
    print()

    # Reset model to initial state (same weights for candidate)
    print("Resetting model to initial state...")
    model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
    print()

    # Run candidate (miner's inner_steps) with gradient capture
    print("Running your inner_steps...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Run all but last step normally
    data_iter = create_iterator()
    if num_steps > 1:
        train_module.inner_steps(model, data_iter, optimizer, num_steps - 1, device)

    # Run final step manually to capture gradients
    batch = next(data_iter)
    batch = batch.to(device, dtype=torch.long)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

    loss.backward()
    candidate_grad = capture_gradients(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # total_tokens computed from controlled test inputs (batch_size * seq_len * num_steps)
    # This mirrors production where validator controls the data iterator
    actual_tokens = batch_size * seq_len * num_steps
    candidate = InnerStepsResult(
        final_logits=logits.detach().float(),
        total_tokens=actual_tokens,
        final_loss=float(loss.item()),
    )

    print(f"  Candidate loss: {candidate.final_loss:.6f}")
    print(f"  Candidate tokens: {candidate.total_tokens:,}")
    print(f"  Candidate gradient norm: {candidate_grad.norm:.4f}")

    # Verify (same checks as production validator)
    passed = verify_all(
        reference=reference,
        candidate=candidate,
        expected_tokens=expected_tokens,
        initial_state=initial_state,
        model=model,
        reference_grad=reference_grad,
        candidate_grad=candidate_grad,
        max_loss_difference=max_loss_difference,
        min_trainable_ratio=min_trainable_ratio,
        min_changed_ratio=min_changed_ratio,
        gradient_cosine_min=gradient_cosine_min,
        gradient_norm_ratio_min=gradient_norm_ratio_min,
        gradient_norm_ratio_max=gradient_norm_ratio_max,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
