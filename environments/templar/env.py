"""
Templar TPS Evaluation Environment (Validator-Owned)

URL-Based Architecture:
- This env.py is owned by the VALIDATOR, not the miner
- Miner commits a URL pointing to their train.py code
- Validator downloads code from URL and passes it to this env
- This ensures miners can't tamper with evaluation logic

Flow:
1. Validator reads miner commitment (contains code URL)
2. Validator downloads train.py from URL
3. Validator calls env.evaluate(code="...", ...)
4. This Actor runs benchmark and returns MFU
"""

import ast
import gc
import hashlib
import importlib.util
import logging
import os
import random
import sys
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel

# Setup logging - only for this module, don't configure root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)

# Suppress noisy loggers from libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)

# Configuration from environment variables
DETERMINISTIC_MODE = os.getenv("DETERMINISTIC_MODE", "1") == "1"
EVAL_SEQUENCE_LENGTH = int(os.getenv("EVAL_SEQUENCE_LENGTH", "1024"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/templar_eval"))


@dataclass
class InnerStepsResult:
    """Expected return type from miner's inner_steps function."""

    final_logits: torch.Tensor
    total_tokens: int
    final_loss: float


@dataclass
class GradientInfo:
    """Captured gradient information for verification.

    MEMORY OPTIMIZATION: grad_vector is a list of per-layer tensors stored on CPU,
    rather than a single concatenated tensor. This avoids allocating ~6-12GB for
    large models. Cosine similarity is computed incrementally layer-by-layer.
    """

    grad_norm: float  # L2 norm of all gradients
    grad_vector: list | None = None  # List of per-layer gradient tensors (on CPU)
    layers_with_grad: int = 0  # Layers with non-zero gradients
    total_layers: int = 0  # Total trainable layers


# Global cache for model (data is NOT cached for validators)
_CACHE = {
    "model": None,
    "model_path": None,
    "initial_state": None,
}


def _load_miner_module(train_path: Path):
    """Dynamically load miner's train.py as a module.

    Args:
        train_path: Path to train.py

    Returns:
        Loaded module with inner_steps function

    Security:
        - Docker provides isolation (--network=none, restricted filesystem)
        - No import sandboxing (incompatible with ML libraries)
    """
    spec = importlib.util.spec_from_file_location("miner_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {train_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["miner_train"] = module
    spec.loader.exec_module(module)
    return module


def _reset_torch_state():
    """Reset torch global state after miner code execution.

    This prevents miners from leaving malicious state that affects
    subsequent evaluations or verification.
    """
    # Reset CUDA state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Reset deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Ensure gradients are enabled
    torch.set_grad_enabled(True)

    # Reset default dtype (miners might change this)
    torch.set_default_dtype(torch.float32)

    logger.debug("Torch state reset complete")


def _load_hf_dataset(
    dataset_name: str,
    model_name: str,
    num_samples: int = 10000,
    sequence_length: int = 1024,
    split: str = "train",
    validator_seed: str | None = None,
) -> torch.Tensor:
    """Load and tokenize dataset from HuggingFace or local cache.

    SECURITY: Validators use unpredictable seeds so miners can't pre-compute
    which samples will be used for evaluation.

    When running with --network none, uses pre-cached dataset from Docker image.
    The validator seed is used to shuffle the cached data unpredictably.

    Args:
        dataset_name: HuggingFace dataset name
        model_name: Model name for tokenizer
        num_samples: Number of samples to load
        sequence_length: Sequence length
        split: Dataset split
        validator_seed: Seed string from validator (for unpredictable sampling)

    Returns:
        Tensor of shape [num_samples, sequence_length]
    """
    import json
    import random

    from transformers import AutoTokenizer

    # Determine seed: validators use unpredictable seed
    if validator_seed:
        seed_hash = hashlib.sha256(validator_seed.encode()).hexdigest()
        actual_seed = int(seed_hash[:8], 16)
        logger.info(f"Validator mode: seed={actual_seed} (from {validator_seed})")
    else:
        actual_seed = 42
        logger.info(f"Test mode: fixed seed={actual_seed}")

    logger.info(f"Loading dataset: {dataset_name} (samples={num_samples})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for cached dataset (enables --network none operation)
    cached_path = os.getenv("CACHED_DATASET_PATH", "/home/appuser/.cache/templar/dataset.json")

    if Path(cached_path).exists():
        # Load from cache and shuffle with validator seed
        logger.info(f"Using cached dataset: {cached_path}")
        with open(cached_path) as f:
            all_samples = json.load(f)

        # Shuffle with validator seed for unpredictability
        rng = random.Random(actual_seed)
        rng.shuffle(all_samples)

        tokens_list = []
        for text in all_samples[:num_samples]:
            encoded = tokenizer(
                text,
                max_length=sequence_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokens_list.append(encoded["input_ids"].squeeze(0))

        if not tokens_list:
            raise ValueError("No samples in cached dataset")

        data = torch.stack(tokens_list)
        logger.info(f"Loaded cached data: shape={data.shape}, seed={actual_seed}")
        return data

    # Fallback: Load from HuggingFace (requires network)
    logger.info("No cache found, loading from HuggingFace...")
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    dataset = dataset.shuffle(seed=actual_seed, buffer_size=10000)

    tokens_list = []
    dataset_iter = iter(dataset)

    for _ in range(num_samples):
        try:
            sample = next(dataset_iter)
            text = sample.get("text", sample.get("content", ""))

            encoded = tokenizer(
                text,
                max_length=sequence_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tokens_list.append(encoded["input_ids"].squeeze(0))

        except StopIteration:
            break

    if not tokens_list:
        raise ValueError(f"No samples loaded from {dataset_name}")

    data = torch.stack(tokens_list)
    logger.info(f"Loaded data: shape={data.shape}, seed={actual_seed}")
    return data


def _set_deterministic(seed: int) -> None:
    """Set deterministic mode for reproducibility."""
    if not DETERMINISTIC_MODE:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _validate_code_structure(code: str) -> tuple[bool, str | None]:
    """Validate that train.py has correct structure."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error at line {exc.lineno}: {exc.msg}"

    inner_steps_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "inner_steps":
            inner_steps_found = True
            args = node.args
            if len(args.args) < 5:
                return False, f"inner_steps has {len(args.args)} args, expected at least 5"
            break

    if not inner_steps_found:
        return False, "Missing required function: inner_steps"

    return True, None


def _validate_return_type(result) -> tuple[bool, str | None, InnerStepsResult | None]:
    """Validate that inner_steps returned correct type."""
    if isinstance(result, InnerStepsResult):
        return True, None, result

    if all(hasattr(result, attr) for attr in ("final_logits", "total_tokens", "final_loss")):
        return (
            True,
            None,
            InnerStepsResult(
                final_logits=result.final_logits,
                total_tokens=result.total_tokens,
                final_loss=result.final_loss,
            ),
        )

    return False, f"Invalid return type from inner_steps: {type(result)}", None


def _load_model(model_path: str, use_random_init: bool = False):
    """Load model from HuggingFace.

    Args:
        model_path: HuggingFace model name/path
        use_random_init: If True, initialize with random weights (anti-cheat)

    Note:
        Docker evaluation runs with --network=none for security.
        Models must be pre-cached in the Docker image. If cache is missing,
        rebuild the image: docker build --network=host -f environments/templar/Dockerfile -t templar-eval:latest .
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    if use_random_init:
        # Random init - miners can't cheat by freezing pretrained layers
        logger.info(f"Loading model config from {model_path} with RANDOM initialization")

        # Use local cache only - Docker runs with --network=none for security
        # If this fails, the Docker image needs to be rebuilt with the model cached
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        # Simple model loading - Qwen2.5-3B (~6GB) fits easily on A100 80GB
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = model.to(device)
        logger.info(f"Model loaded on {device}")
    else:
        logger.info(f"Loading pretrained model from {model_path}")
        # Use local cache only - Docker runs with --network=none for security
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
            local_files_only=True,
        )

    model.train()
    
    # Ensure ALL parameters have requires_grad=True for training
    for param in model.parameters():
        param.requires_grad = True
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def _count_model_params(model: torch.nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def _calculate_mfu(
    total_tokens: int,
    wall_time: float,
    model_params: int,
    gpu_peak_tflops: float = 312.0,
) -> float:
    """Calculate Model FLOPs Utilization (MFU).

    MFU = actual_flops / theoretical_peak_flops

    For transformer training (forward + backward):
    - Forward: 2 * params * tokens (multiply-accumulate)
    - Backward: 4 * params * tokens (gradients are 2x forward)
    - Total: 6 * params * tokens

    Args:
        total_tokens: Total tokens processed
        wall_time: Wall clock time in seconds
        model_params: Number of model parameters
        gpu_peak_tflops: GPU theoretical peak TFLOPS (A100 80GB = 312 TFLOPS bfloat16)

    Returns:
        MFU as a percentage (0-100)
    """
    # Guard against division by zero / invalid inputs
    if wall_time <= 0 or gpu_peak_tflops <= 0:
        return 0.0

    # FLOPs for forward + backward pass
    flops_per_token = 6 * model_params
    total_flops = flops_per_token * total_tokens

    # Actual TFLOPS achieved
    actual_tflops = total_flops / wall_time / 1e12

    # MFU as percentage
    mfu = (actual_tflops / gpu_peak_tflops) * 100

    return min(mfu, 100.0)  # Cap at 100%


def _capture_gradients(model: torch.nn.Module) -> GradientInfo:
    """Capture gradient information from model after backward pass.

    MEMORY OPTIMIZATION: Instead of concatenating all gradients into a single
    large tensor (which can be ~6-12GB for 3B models), we store per-layer
    gradient vectors on CPU. Cosine similarity is computed incrementally
    in _verify_gradients() to avoid memory pressure.

    Returns:
        GradientInfo with norm and per-layer gradient vectors (on CPU)
    """
    grad_vectors_cpu = []  # Store per-layer on CPU to save GPU memory
    total_norm_sq = 0.0
    layers_with_grad = 0
    layers_without_grad = 0

    for param in model.parameters():
        if param.grad is not None:
            # Move to CPU immediately to free GPU memory
            grad_flat = param.grad.detach().cpu().float().view(-1)
            total_norm_sq += grad_flat.pow(2).sum().item()
            # Check if gradient is actually non-zero (not just allocated)
            if grad_flat.abs().sum().item() > 1e-10:
                layers_with_grad += 1
                grad_vectors_cpu.append(grad_flat)
            else:
                layers_without_grad += 1
                grad_vectors_cpu.append(grad_flat)  # Still store for shape matching
        else:
            layers_without_grad += 1
            grad_vectors_cpu.append(None)

    grad_norm = total_norm_sq**0.5

    # Log layer gradient coverage
    total_layers = layers_with_grad + layers_without_grad
    if total_layers > 0:
        coverage = layers_with_grad / total_layers
        logger.info(f"Gradient coverage: {layers_with_grad}/{total_layers} layers ({coverage:.1%})")
        if layers_without_grad > 0:
            logger.warning(f"WARNING: {layers_without_grad} layers have zero/no gradients!")

    return GradientInfo(
        grad_norm=grad_norm,
        grad_vector=grad_vectors_cpu,  # Now a list of per-layer tensors on CPU
        layers_with_grad=layers_with_grad,
        total_layers=total_layers,
    )


def _verify_trainable_params(
    model: torch.nn.Module,
    min_trainable_ratio: float = 1.0,
) -> tuple[bool, str | None, dict]:
    """Check that minimum % of params are trainable (prevents layer freezing)."""
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0

    details = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_ratio,
        "min_required": min_trainable_ratio,
    }

    if trainable_ratio < min_trainable_ratio:
        error = (
            f"Insufficient trainable params: {trainable_ratio:.1%} "
            f"({trainable_params:,}/{total_params:,}) - minimum {min_trainable_ratio:.0%} required"
        )
        logger.error(f"[FAILED] {error}")
        return False, error, details

    logger.info(
        f"[PASSED] Trainable params check ({trainable_ratio:.1%} >= {min_trainable_ratio:.0%})"
    )
    return True, None, details


def _verify_params_changed(
    model: torch.nn.Module,
    initial_state: dict,
    min_changed_ratio: float = 0.5,
    element_threshold: float = 1e-6,
) -> tuple[bool, str | None, dict]:
    """Verify that minimum % of individual parameter elements changed during training.

    SECURITY: Counts individual elements, not tensors. This prevents a bypass where
    miners make tiny modifications to many tensors (each barely above threshold) to
    pass the check without meaningful training.

    Args:
        model: Model after training
        initial_state: Model state before training
        min_changed_ratio: Minimum fraction of elements that must change
        element_threshold: Minimum absolute change for an element to count as "changed"
    """
    total_elements = 0
    changed_elements = 0

    for name, param in model.named_parameters():
        if name in initial_state:
            initial = initial_state[name].to(param.device)
            # Count individual elements that changed (not whole tensors)
            element_diffs = (param.data - initial).abs()
            changed_mask = element_diffs > element_threshold

            total_elements += param.numel()
            changed_elements += changed_mask.sum().item()

    changed_ratio = changed_elements / total_elements if total_elements > 0 else 0.0

    details = {
        "total_elements": total_elements,
        "changed_elements": int(changed_elements),
        "changed_ratio": changed_ratio,
        "min_required": min_changed_ratio,
        "element_threshold": element_threshold,
    }

    if changed_ratio < min_changed_ratio:
        error = (
            f"Insufficient parameter updates: {changed_ratio:.1%} "
            f"({int(changed_elements):,}/{total_elements:,} elements) - minimum {min_changed_ratio:.0%} required"
        )
        logger.error(f"[FAILED] {error}")
        return False, error, details

    logger.info(
        f"[PASSED] Parameter changes check ({changed_ratio:.1%} >= {min_changed_ratio:.0%})"
    )
    return True, None, details


def _get_cached_model(model_path: str, use_random_init: bool = False):
    """Get model from cache or load it."""
    cached = _CACHE.get("model")
    cached_path = _CACHE.get("model_path")
    cached_random_init = _CACHE.get("use_random_init")

    # Cache hit only if path AND init mode match
    if cached is not None and cached_path == model_path and cached_random_init == use_random_init:
        if _CACHE.get("initial_state"):
            cached.load_state_dict(_CACHE["initial_state"])
        return cached

    model = _load_model(model_path, use_random_init=use_random_init)
    _CACHE["model"] = model
    _CACHE["model_path"] = model_path
    _CACHE["use_random_init"] = use_random_init
    _CACHE["initial_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model


def _create_data_iterator(
    data: torch.Tensor, batch_size: int, sequence_length: int
) -> Iterator[torch.Tensor]:
    """Create infinite data iterator."""
    if data.size(1) < sequence_length:
        raise ValueError(f"Data sequence length {data.size(1)} < required {sequence_length}")

    data = data[:, :sequence_length]
    num_samples = data.size(0)

    def _iter():
        idx = 0
        while True:
            end_idx = idx + batch_size
            if end_idx > num_samples:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx

    return _iter()


def _create_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    """Create standard AdamW optimizer."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )


class GradientCapturingOptimizer:
    """Optimizer wrapper that captures gradients before each step.

    SECURITY: This wrapper intercepts optimizer.step() to capture gradients
    from within the miner's inner_steps execution. This ensures we verify
    gradients produced by miner code, not validator code.

    The previous approach (running final step manually outside inner_steps)
    was flawed because when steps==1, the miner's inner_steps was never
    called during full evaluation, only during warmup.

    PERFORMANCE: Tracks overhead time separately and excludes it from wall_time
    for fair MFU/TPS calculation. MFU formula (6 * params * tokens) only accounts
    for forward + backward FLOPs, so we exclude:
    - Gradient capture (validator overhead)

    """

    def __init__(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module):
        self.optimizer = optimizer
        self.model = model
        self.captured_gradients: GradientInfo | None = None
        self.step_count = 0
        self.gradient_capture_time: float = 0.0  # Cumulative time spent in gradient capture

    def step(self, *args, **kwargs):
        """Capture gradients BEFORE optimizer.step() clears them."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Now time ONLY the gradient capture (GPU→CPU copy for verification)
        # This is validator overhead and should be excluded from wall_time
        capture_start = time.perf_counter()
        self.captured_gradients = _capture_gradients(self.model)
        self.gradient_capture_time += time.perf_counter() - capture_start

        # Actual optimizer step (part of miner's training)
        self.step_count += 1
        return self.optimizer.step(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = False):
        """Forward to underlying optimizer."""
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        """Forward param_groups access to underlying optimizer."""
        return self.optimizer.param_groups

    @param_groups.setter
    def param_groups(self, value):
        """Forward param_groups setter to underlying optimizer."""
        self.optimizer.param_groups = value

    def state_dict(self):
        """Forward to underlying optimizer."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Forward to underlying optimizer."""
        return self.optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        """Forward to underlying optimizer."""
        return self.optimizer.add_param_group(param_group)

    @property
    def state(self):
        """Forward state access to underlying optimizer."""
        return self.optimizer.state

    def __getattr__(self, name):
        """Forward any other attribute access to underlying optimizer."""
        return getattr(self.optimizer, name)


def _run_reference(
    model: torch.nn.Module,
    data_iterator: Iterator[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    device: torch.device,
    capture_final_gradients: bool = True,
) -> tuple[InnerStepsResult, GradientInfo | None]:
    """Run reference implementation for comparison.

    Returns:
        Tuple of (InnerStepsResult, GradientInfo) where GradientInfo
        contains the gradients from the final step (before optimizer.step)
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    total_tokens = 0
    final_logits = None
    final_loss = 0.0
    final_gradients = None

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

        # Capture gradients on final step (before optimizer.step clears them)
        if capture_final_gradients and step == num_steps - 1:
            final_gradients = _capture_gradients(model)

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
    return result, final_gradients


def _verify_gradients(
    reference_grad: GradientInfo | None,
    candidate_grad: GradientInfo | None,
    cosine_min: float = 0.8,
    norm_ratio_min: float = 0.5,
    norm_ratio_max: float = 2.0,
) -> tuple[bool, str | None, dict]:
    """Verify candidate gradients match reference using cosine similarity and norm ratio.

    Args:
        reference_grad: Reference implementation gradients
        candidate_grad: Miner's implementation gradients
        cosine_min: Minimum cosine similarity required
        norm_ratio_min: Minimum ratio of candidate/reference gradient norm
        norm_ratio_max: Maximum ratio of candidate/reference gradient norm

    Returns:
        Tuple of (success, error_message, details)
    """
    details = {
        "cosine_min": cosine_min,
        "norm_ratio_min": norm_ratio_min,
        "norm_ratio_max": norm_ratio_max,
        "checks_passed": [],
        "checks_failed": [],
    }

    logger.info("=" * 60)
    logger.info("VERIFICATION: Gradient-based verification")
    logger.info("=" * 60)

    if reference_grad is None or candidate_grad is None:
        error = "Missing gradient information for verification"
        details["checks_failed"].append({"check": "gradient_availability", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    details["reference_grad_norm"] = reference_grad.grad_norm
    details["candidate_grad_norm"] = candidate_grad.grad_norm
    details["candidate_layers_with_grad"] = candidate_grad.layers_with_grad
    details["candidate_total_layers"] = candidate_grad.total_layers

    # Check 0: All layers must have gradients (catches "restore weights" attack)
    if candidate_grad.total_layers > 0:
        grad_coverage = candidate_grad.layers_with_grad / candidate_grad.total_layers
        details["gradient_coverage"] = grad_coverage
        logger.info(f"[CHECK 0/3] Gradient coverage: {grad_coverage:.1%}")
        logger.info(
            f"   Layers with gradients: {candidate_grad.layers_with_grad}/{candidate_grad.total_layers}"
        )

        if grad_coverage < 1.0:
            error = (
                f"Not all layers have gradients: {candidate_grad.layers_with_grad}/{candidate_grad.total_layers} "
                f"({grad_coverage:.1%}) - possible layer freezing detected"
            )
            details["checks_failed"].append({"check": "gradient_coverage", "error": error})
            details["error_code"] = "gradient_coverage_failed"
            logger.error(f"[FAILED] {error}")
            return False, error, details
        details["checks_passed"].append("gradient_coverage")
        logger.info("[PASSED] All layers have gradients")

    # Check 1: Gradient norm ratio
    if reference_grad.grad_norm > 0:
        norm_ratio = candidate_grad.grad_norm / reference_grad.grad_norm
        details["norm_ratio"] = norm_ratio
        logger.info(f"[CHECK 1/3] Gradient norm ratio: {norm_ratio:.4f}")
        logger.info(f"   Reference norm: {reference_grad.grad_norm:.6f}")
        logger.info(f"   Candidate norm: {candidate_grad.grad_norm:.6f}")
        logger.info(f"   Allowed range: [{norm_ratio_min}, {norm_ratio_max}]")

        if norm_ratio < norm_ratio_min or norm_ratio > norm_ratio_max:
            error = (
                f"Gradient norm ratio {norm_ratio:.4f} outside allowed range "
                f"[{norm_ratio_min}, {norm_ratio_max}]"
            )
            details["checks_failed"].append({"check": "gradient_norm_ratio", "error": error})
            details["error_code"] = "gradient_norm_ratio_failed"
            logger.error(f"[FAILED] {error}")
            return False, error, details
        details["checks_passed"].append("gradient_norm_ratio")
        logger.info("[PASSED] Gradient norm ratio in acceptable range")
    else:
        logger.warning("Reference gradient norm is zero, skipping norm ratio check")

    # Check 2: Cosine similarity (computed incrementally to save memory)
    ref_vecs = reference_grad.grad_vector  # List of per-layer tensors
    cand_vecs = candidate_grad.grad_vector  # List of per-layer tensors

    if ref_vecs is not None and cand_vecs is not None and len(ref_vecs) > 0:
        # Verify same number of layers
        if len(ref_vecs) != len(cand_vecs):
            error = f"Gradient layer count mismatch: ref={len(ref_vecs)}, cand={len(cand_vecs)}"
            details["checks_failed"].append({"check": "gradient_shape", "error": error})
            logger.error(f"[FAILED] {error}")
            return False, error, details

        # Compute cosine similarity incrementally (layer-by-layer)
        # cosine_sim = dot_product / (norm_ref * norm_cand)
        # We already have norms from GradientInfo, just need dot product
        dot_product = 0.0
        total_elements = 0

        for ref_layer, cand_layer in zip(ref_vecs, cand_vecs):
            if ref_layer is None or cand_layer is None:
                continue

            # Verify shape match for this layer
            if ref_layer.shape != cand_layer.shape:
                error = f"Gradient shape mismatch at layer: ref={ref_layer.shape}, cand={cand_layer.shape}"
                details["checks_failed"].append({"check": "gradient_shape", "error": error})
                logger.error(f"[FAILED] {error}")
                return False, error, details

            # Accumulate dot product (both tensors are on CPU)
            dot_product += (ref_layer * cand_layer).sum().item()
            total_elements += ref_layer.numel()

        # Calculate cosine similarity using pre-computed norms
        ref_norm = reference_grad.grad_norm
        cand_norm = candidate_grad.grad_norm

        if ref_norm > 0 and cand_norm > 0:
            cosine_sim = dot_product / (ref_norm * cand_norm)
        else:
            cosine_sim = 0.0

        details["cosine_similarity"] = cosine_sim
        details["gradient_elements"] = total_elements

        logger.info(f"[CHECK 2/3] Gradient cosine similarity: {cosine_sim:.4f}")
        logger.info(f"   Minimum required: {cosine_min}")
        logger.info(f"   Total gradient elements: {total_elements:,}")

        if cosine_sim < cosine_min:
            error = f"Gradient cosine similarity {cosine_sim:.4f} below minimum {cosine_min}"
            details["checks_failed"].append({"check": "gradient_cosine", "error": error})
            details["error_code"] = "gradient_cosine_failed"
            logger.error(f"[FAILED] {error}")
            return False, error, details
        details["checks_passed"].append("gradient_cosine_similarity")
        logger.info("[PASSED] Gradient cosine similarity acceptable")
    else:
        logger.warning("Gradient vectors unavailable, skipping cosine similarity check")

    logger.info("=" * 60)
    logger.info("VERIFICATION: GRADIENT CHECKS PASSED")
    logger.info(f"   Checks passed: {details['checks_passed']}")
    logger.info("=" * 60)

    return True, None, details


def _verify_outputs(
    reference: InnerStepsResult,
    candidate: InnerStepsResult,
    expected_tokens: int,
    reference_grad: GradientInfo | None = None,
    candidate_grad: GradientInfo | None = None,
    max_loss_difference: float = 0.5,
    gradient_cosine_min: float = 0.8,
    gradient_norm_ratio_min: float = 0.5,
    gradient_norm_ratio_max: float = 2.0,
) -> tuple[bool, str | None, dict]:
    """Verify candidate outputs match reference.

    Verification checks:
    1. Token count matches expected
    2. Loss is valid and similar to reference (small difference)
    3. Gradient-based verification (replaces logits comparison)

    Args:
        reference: Reference implementation results
        candidate: Miner's implementation results
        expected_tokens: Expected token count
        reference_grad: Reference gradients for comparison
        candidate_grad: Candidate gradients for comparison
        max_loss_difference: Maximum allowed |candidate_loss - reference_loss|
        gradient_cosine_min: Minimum gradient cosine similarity
        gradient_norm_ratio_min: Min gradient norm ratio
        gradient_norm_ratio_max: Max gradient norm ratio

    Returns:
        Tuple of (success, error_message, verification_details)
    """
    details = {
        "expected_tokens": expected_tokens,
        "candidate_tokens": candidate.total_tokens,
        "candidate_loss": candidate.final_loss,
        "reference_loss": reference.final_loss if reference else None,
        "checks_passed": [],
        "checks_failed": [],
    }

    logger.info("=" * 60)
    logger.info("VERIFICATION: Starting output verification")
    logger.info("=" * 60)

    # 1. Verify token count matches expected
    logger.info(
        f"[CHECK 1/3] Token count: expected={expected_tokens}, got={candidate.total_tokens}"
    )
    if candidate.total_tokens != expected_tokens:
        error = f"Token count mismatch: expected {expected_tokens}, got {candidate.total_tokens}"
        details["checks_failed"].append({"check": "token_count", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    details["checks_passed"].append("token_count")
    logger.info("[PASSED] Token count matches")

    # 2. Verify loss is reasonable and similar to reference
    logger.info(f"[CHECK 2/3] Loss validity: candidate_loss={candidate.final_loss:.6f}")
    if candidate.final_loss != candidate.final_loss:  # NaN check
        error = "Loss is NaN"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details
    if abs(candidate.final_loss) > 100:
        error = f"Loss unreasonable: {candidate.final_loss:.4f} (expected 1-10)"
        details["checks_failed"].append({"check": "loss_validity", "error": error})
        logger.error(f"[FAILED] {error}")
        return False, error, details

    # Compare losses - check absolute difference
    if reference is not None and reference.final_loss > 0:
        loss_difference = abs(candidate.final_loss - reference.final_loss)
        details["loss_difference"] = loss_difference
        logger.info(
            f"   Reference loss: {reference.final_loss:.4f}, "
            f"Candidate loss: {candidate.final_loss:.4f}"
        )
        logger.info(
            f"   Loss difference: {loss_difference:.4f} (max allowed: {max_loss_difference})"
        )

        if loss_difference > max_loss_difference:
            error = (
                f"Loss difference too large: candidate={candidate.final_loss:.4f}, "
                f"reference={reference.final_loss:.4f}, diff={loss_difference:.4f} > {max_loss_difference}"
            )
            details["checks_failed"].append({"check": "loss_comparison", "error": error})
            logger.error(f"[FAILED] {error}")
            return False, error, details
    details["checks_passed"].append("loss_validity")
    logger.info("[PASSED] Loss is valid and similar to reference")

    # 3. Gradient-based verification (replaces logits comparison)
    logger.info("[CHECK 3/3] Gradient verification")
    grad_ok, grad_error, grad_details = _verify_gradients(
        reference_grad,
        candidate_grad,
        cosine_min=gradient_cosine_min,
        norm_ratio_min=gradient_norm_ratio_min,
        norm_ratio_max=gradient_norm_ratio_max,
    )
    details["gradient_verification"] = grad_details

    if not grad_ok:
        details["checks_failed"].append({"check": "gradient_verification", "error": grad_error})
        return False, grad_error, details
    details["checks_passed"].append("gradient_verification")

    logger.info("=" * 60)
    logger.info("VERIFICATION: ALL CHECKS PASSED")
    logger.info(f"   Checks passed: {details['checks_passed']}")
    logger.info("=" * 60)

    return True, None, details


class Actor:
    """Templar MFU Evaluation Actor for Affinetes (Validator-Owned).

    This Actor is owned by the validator, not the miner.
    Code is passed directly from the validator (downloaded from URL).
    """

    async def evaluate(
        self,
        task_id: int = 0,
        seed: str = "0:0:0",
        model_url: str = "",
        data_url: str = "",
        steps: int = 5,
        batch_size: int = 8,
        timeout: int = 600,
        sequence_length: int | None = None,
        data_samples: int = 10000,
        code: str = "",  # Miner's code passed directly
        # Verification settings
        max_loss_difference: float = 0.5,
        use_random_init: bool = True,
        min_trainable_params_ratio: float = 1.0,
        min_params_changed_ratio: float = 0.5,
        # Gradient verification (replaces logits)
        gradient_cosine_min: float = 0.8,
        gradient_norm_ratio_min: float = 0.5,
        gradient_norm_ratio_max: float = 2.0,
        # MFU calculation
        gpu_peak_tflops: float = 312.0,
        model_params_override: int | None = None,
    ) -> dict:
        """
        Run MFU evaluation on miner's code.

        Args:
            task_id: Evaluation run identifier
            seed: Deterministic seed string (format: "block:uid:run_idx")
            model_url: HuggingFace model name
            data_url: HuggingFace dataset name
            steps: Number of training steps
            batch_size: Batch size
            timeout: Maximum seconds
            sequence_length: Sequence length
            data_samples: Number of data samples
            code: Miner's train.py code (passed directly from validator)
            max_loss_difference: Max allowed |candidate_loss - reference_loss|
            use_random_init: Use random weights (anti-cheat)
            min_trainable_params_ratio: Min % params that must be trainable
            min_params_changed_ratio: Min % params that must change
            gradient_cosine_min: Min gradient cosine similarity
            gradient_norm_ratio_min: Min gradient norm ratio
            gradient_norm_ratio_max: Max gradient norm ratio
            gpu_peak_tflops: GPU peak TFLOPS for MFU calculation
            model_params_override: Override model param count (None = auto-detect)

        Returns:
            Dict with: mfu, tps, total_tokens, wall_time_seconds, success, error, code
        """
        # Validate code
        if not code:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "No code provided",
                "seed": seed,
            }

        # Validate model/data
        if not model_url or not data_url:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "Missing model_url or data_url",
                "seed": seed,
            }

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout
        seed_value = abs(hash(seed)) % (2**32)
        _set_deterministic(seed_value)

        # Validate code structure
        code_ok, code_error = _validate_code_structure(code)
        if not code_ok:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": code_error,
                "seed": seed,
                "code": code,
            }

        # Write code to temp file for loading as module
        train_path = CACHE_DIR / "miner_train.py"
        try:
            train_path.write_text(code)
        except Exception as e:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": f"Failed to write train.py: {e}",
                "seed": seed,
            }

        if time.monotonic() > deadline:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": "Timeout before evaluation",
                "seed": seed,
            }

        try:
            # Load miner's module
            miner_module = _load_miner_module(train_path)

            if not hasattr(miner_module, "inner_steps"):
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": "train.py missing inner_steps function",
                    "seed": seed,
                    "code": code,
                }

            # Load model and data (with random init if enabled for anti-cheat)
            model = _get_cached_model(model_url, use_random_init=use_random_init)
            model_params = model_params_override or _count_model_params(model)
            logger.info(f"Model loaded: {model_params:,} parameters, random_init={use_random_init}")

            data = _load_hf_dataset(
                dataset_name=data_url,
                model_name=model_url,
                num_samples=data_samples,
                sequence_length=sequence_length or EVAL_SEQUENCE_LENGTH,
                validator_seed=seed,  # Unpredictable sampling
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            seq_len = sequence_length or EVAL_SEQUENCE_LENGTH

            # Run reference implementation (captures gradients for verification)
            data_iter_ref = _create_data_iterator(data, batch_size, seq_len)
            optimizer_ref = _create_optimizer(model)

            initial_state = _CACHE.get("initial_state")
            if initial_state is None:
                initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                _CACHE["initial_state"] = initial_state

            reference, reference_grad = _run_reference(
                model, data_iter_ref, optimizer_ref, steps, device, capture_final_gradients=True
            )

            # Reset model for miner's code
            model.load_state_dict(initial_state)
            warmup_steps = 2
            data_iter_warmup = _create_data_iterator(data, batch_size, seq_len)
            optimizer_warmup = _create_optimizer(model)

            logger.info(f"Running {warmup_steps} warmup step(s) to check for basic errors...")
            try:
                warmup_result = miner_module.inner_steps(
                    model=model,
                    data_iterator=data_iter_warmup,
                    optimizer=optimizer_warmup,
                    num_steps=warmup_steps,
                    device=device,
                )

                # Quick validation of warmup result
                if warmup_result is None:
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": "inner_steps returned None (warmup check)",
                        "seed": seed,
                        "code": code,
                    }

                # Check required attributes exist
                if not hasattr(warmup_result, "final_logits"):
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": "inner_steps result missing 'final_logits' attribute",
                        "seed": seed,
                        "code": code,
                    }

                # Check logits shape (should be 3D: batch, seq, vocab)
                logits = warmup_result.final_logits
                if logits is not None and len(logits.shape) != 3:
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": f"Logits shape mismatch: expected 3D (batch, seq, vocab), got {logits.shape}",
                        "seed": seed,
                        "code": code,
                    }

                logger.info("Warmup passed - proceeding with verification checks")

                # Check trainable params AFTER warmup (miner code may modify requires_grad)
                trainable_ok, trainable_error, trainable_details = _verify_trainable_params(
                    model, min_trainable_params_ratio
                )
                if not trainable_ok:
                    return {
                        "task_id": task_id,
                        "mfu": 0.0,
                        "tps": 0.0,
                        "total_tokens": 0,
                        "wall_time_seconds": 0.0,
                        "success": False,
                        "error": trainable_error,
                        "error_code": "insufficient_trainable_params",
                        "seed": seed,
                        "code": code,
                        "diagnostics": {"trainable_params": trainable_details},
                    }

            except Exception as e:
                # Early termination - don't waste time on broken code
                error_msg = f"Early termination (warmup failed): {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": 0.0,
                    "success": False,
                    "error": error_msg,
                    "seed": seed,
                    "code": code,
                }

            # Reset model state after warmup
            model.load_state_dict(initial_state)

            # =================================================================
            # FULL EVALUATION - Run actual timed evaluation with gradient capture
            # =================================================================
            # SECURITY: Use GradientCapturingOptimizer to capture gradients from
            # WITHIN the miner's inner_steps execution. This ensures we verify
            # gradients produced by miner code, not validator code.
            #
            # Previous approach (running final step manually) was flawed because
            # when steps==1, miner's inner_steps was never called during evaluation.
            data_iter_miner = _create_data_iterator(data, batch_size, seq_len)
            base_optimizer = _create_optimizer(model)
            optimizer_miner = GradientCapturingOptimizer(base_optimizer, model)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            # Run ALL steps through miner's inner_steps
            miner_result = miner_module.inner_steps(
                model=model,
                data_iterator=data_iter_miner,
                optimizer=optimizer_miner,
                num_steps=steps,
                device=device,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_time = time.perf_counter() - start

            # Exclude gradient capture time from wall_time for fair MFU calculation
            # Gradient capture is validator overhead (copies gradients GPU→CPU for verification)
            # optimizer.step() and zero_grad() ARE part of miner's training, included in wall_time
            gradient_capture_time = optimizer_miner.gradient_capture_time
            wall_time = total_time - gradient_capture_time
            logger.info(
                f"Timing: total={total_time:.2f}s, grad_capture={gradient_capture_time:.2f}s, "
                f"wall_time={wall_time:.2f}s (used for MFU/TPS)"
            )

            # Get gradients captured from the final step (inside miner's code)
            candidate_grad = optimizer_miner.captured_gradients

            if candidate_grad is None:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": "No gradients captured - miner may not be calling optimizer.step()",
                    "error_code": "no_gradients_captured",
                    "seed": seed,
                    "code": code,
                }

            # Validate miner's returned result
            ok, error, parsed = _validate_return_type(miner_result)
            if not ok or parsed is None:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": error,
                    "seed": seed,
                    "code": code,
                }

            # =================================================================
            # PARAMS CHANGED CHECK - Verify miner actually trained the model
            # =================================================================
            # SECURITY: This catches the "freeze layers then restore requires_grad" attack.
            # Even if a miner restores requires_grad=True after their inner_steps,
            # the frozen layers wouldn't have changed during training, so this check fails.
            params_ok, params_error, params_details = _verify_params_changed(
                model, initial_state, min_params_changed_ratio
            )
            if not params_ok:
                return {
                    "task_id": task_id,
                    "mfu": 0.0,
                    "tps": 0.0,
                    "total_tokens": 0,
                    "wall_time_seconds": wall_time,
                    "success": False,
                    "error": params_error,
                    "error_code": "insufficient_params_changed",
                    "seed": seed,
                    "code": code,
                    "diagnostics": {"params_changed": params_details},
                }

            # Calculate expected tokens
            expected_tokens = batch_size * seq_len * steps

            # Verify outputs using gradient-based verification
            verified, verify_error, verify_details = _verify_outputs(
                reference,
                parsed,
                expected_tokens,
                reference_grad=reference_grad,
                candidate_grad=candidate_grad,
                max_loss_difference=max_loss_difference,
                gradient_cosine_min=gradient_cosine_min,
                gradient_norm_ratio_min=gradient_norm_ratio_min,
                gradient_norm_ratio_max=gradient_norm_ratio_max,
            )

            # Calculate MFU and TPS
            total_tokens_int = int(parsed.total_tokens)
            tps = float(total_tokens_int) / max(wall_time, 1e-6)
            mfu = _calculate_mfu(total_tokens_int, wall_time, model_params, gpu_peak_tflops)

            # Diagnostics
            diagnostics = {
                "verification": verify_details,
                "reference_loss": reference.final_loss,
                "candidate_loss": parsed.final_loss,
                "expected_tokens": expected_tokens,
                "actual_tokens": parsed.total_tokens,
                "model_params": model_params,
                "gpu_peak_tflops": gpu_peak_tflops,
                "trainable_params": trainable_details,
                "params_changed": params_details,
            }

            # Extract error_code from verification details if present
            error_code = None
            if not verified and verify_details:
                # Check gradient_verification sub-details
                grad_details = verify_details.get("gradient_verification", {})
                error_code = grad_details.get("error_code")
                # Check other verification failures
                if not error_code:
                    for check in verify_details.get("checks_failed", []):
                        if check.get("check") == "token_count":
                            error_code = "token_count_mismatch"
                        elif check.get("check") == "loss_comparison":
                            error_code = "loss_mismatch"

            return {
                "task_id": task_id,
                "mfu": mfu if verified else 0.0,
                "tps": tps if verified else 0.0,
                "total_tokens": total_tokens_int if verified else 0,
                "wall_time_seconds": wall_time,
                "success": verified,
                "error": verify_error,
                "error_code": error_code,
                "seed": seed,
                "diagnostics": diagnostics,
                "code": code,
            }

        except Exception as e:
            return {
                "task_id": task_id,
                "mfu": 0.0,
                "tps": 0.0,
                "total_tokens": 0,
                "wall_time_seconds": 0.0,
                "success": False,
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                "seed": seed,
                "code": code,
            }

        finally:
            # SECURITY: Reset torch state to prevent cross-evaluation contamination
            _reset_torch_state()

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# =============================================================================
# FastAPI HTTP Server (for Basilica custom Docker deployment)
# =============================================================================

app = FastAPI(title="Templar MFU Evaluation", version="2.0.0")

# Global actor instance (reused for efficiency)
_actor: Actor | None = None


def get_actor() -> Actor:
    """Get or create singleton Actor instance."""
    global _actor
    if _actor is None:
        _actor = Actor()
    return _actor


class EvaluateRequest(BaseModel):
    """Request body for /evaluate endpoint."""

    task_id: int = 0
    seed: str = "0:0:0"
    model_url: str
    data_url: str
    steps: int = 5
    batch_size: int = 8
    timeout: int = 600
    sequence_length: int | None = None
    data_samples: int = 10000
    code: str  # Miner's train.py code
    # Verification settings
    max_loss_difference: float = 0.5
    use_random_init: bool = True
    min_trainable_params_ratio: float = 1.0
    min_params_changed_ratio: float = 0.5
    # Gradient verification
    gradient_cosine_min: float = 0.8
    gradient_norm_ratio_min: float = 0.5
    gradient_norm_ratio_max: float = 2.0
    # MFU calculation
    gpu_peak_tflops: float = 312.0


class EvaluateResponse(BaseModel):
    """Response body from /evaluate endpoint."""

    task_id: int
    mfu: float  # Model FLOPs Utilization (primary metric)
    tps: float  # Tokens per second (secondary metric)
    total_tokens: int
    wall_time_seconds: float
    success: bool
    error: str | None = None
    seed: str
    diagnostics: dict = {}


@app.get("/health")
async def health():
    """Health check endpoint for Basilica."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate miner's train.py code and return MFU score.

    This endpoint is called by the validator via Basilica.
    """
    actor = get_actor()

    result = await actor.evaluate(
        task_id=request.task_id,
        seed=request.seed,
        model_url=request.model_url,
        data_url=request.data_url,
        steps=request.steps,
        batch_size=request.batch_size,
        timeout=request.timeout,
        sequence_length=request.sequence_length,
        data_samples=request.data_samples,
        code=request.code,
        max_loss_difference=request.max_loss_difference,
        use_random_init=request.use_random_init,
        min_trainable_params_ratio=request.min_trainable_params_ratio,
        min_params_changed_ratio=request.min_params_changed_ratio,
        gradient_cosine_min=request.gradient_cosine_min,
        gradient_norm_ratio_min=request.gradient_norm_ratio_min,
        gradient_norm_ratio_max=request.gradient_norm_ratio_max,
        gpu_peak_tflops=request.gpu_peak_tflops,
    )

    return EvaluateResponse(
        task_id=result.get("task_id", request.task_id),
        mfu=result.get("mfu", 0.0),
        tps=result.get("tps", 0.0),
        total_tokens=result.get("total_tokens", 0),
        wall_time_seconds=result.get("wall_time_seconds", 0.0),
        success=result.get("success", False),
        error=result.get("error"),
        seed=result.get("seed", request.seed),
        diagnostics=result.get("diagnostics", {}),
    )


# Entry point when running directly (for local testing)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
