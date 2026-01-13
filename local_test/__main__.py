#!/usr/bin/env python3
"""
Local testing script for miners.

This allows miners to test their code locally before submitting (and paying).
No network access required, no payment needed.

Usage:
    uv run python -m local_test train.py
"""

import argparse
import ast
import importlib.util
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from tournament.config import get_hparams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def validate_code_structure(code_path: Path) -> tuple[bool, list[str]]:
    """Validate code has required structure.
    
    Args:
        code_path: Path to train.py file
        
    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []
    
    if not code_path.exists():
        return False, [f"File not found: {code_path}"]
    
    code = code_path.read_text()
    
    # Check syntax
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error at line {e.lineno}: {e.msg}"]
    
    # Check for inner_steps function
    has_inner_steps = False
    has_result_class = False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "inner_steps":
            has_inner_steps = True
            # Check parameters
            if len(node.args.args) < 5:
                errors.append("inner_steps must have at least 5 parameters")
        
        if isinstance(node, ast.ClassDef) and node.name == "InnerStepsResult":
            has_result_class = True
    
    if not has_inner_steps:
        errors.append("Missing required function: inner_steps")
    
    if not has_result_class:
        errors.append("Missing required class: InnerStepsResult")
    
    # Check for forbidden imports
    forbidden = ["os", "subprocess", "socket", "http", "urllib", "requests", "pickle"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(fb in alias.name for fb in forbidden):
                    errors.append(f"Forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and any(fb in node.module for fb in forbidden):
                errors.append(f"Forbidden import: {node.module}")
    
    return len(errors) == 0, errors


def run_inner_steps(code_path: Path, num_steps: int = 2) -> tuple[bool, str, float]:
    """Actually run inner_steps to verify it works.
    
    Args:
        code_path: Path to train.py file
        num_steps: Number of steps to run (default: 2 for quick test)
        
    Returns:
        Tuple of (success, message, tps)
    """
    try:
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("train_module", code_path)
        if spec is None or spec.loader is None:
            return False, "Could not load module", 0.0
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["train_module"] = module
        spec.loader.exec_module(module)
        
        # Check for required function
        if not hasattr(module, "inner_steps"):
            return False, "inner_steps function not found", 0.0
        
        inner_steps = module.inner_steps
        
        # Check for benchmark data
        model_path = Path("benchmark/model")
        data_path = Path("benchmark/data/train.pt")
        
        if not model_path.exists():
            return False, f"Model not found at {model_path}. Run: uv run python scripts/setup_benchmark.py", 0.0
        
        if not data_path.exists():
            return False, f"Data not found at {data_path}. Run: uv run python scripts/setup_benchmark.py", 0.0
        
        # Load hparams to match validator exactly
        hparams = get_hparams()
        batch_size = hparams.benchmark_batch_size
        sequence_length = hparams.benchmark_sequence_length
        
        # Check GPU requirements from sandbox config
        required_gpus = hparams.sandbox.gpu_count
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if available_gpus == 0:
            logger.warning("⚠️  No GPU available! Running on CPU (will be slow)")
        elif available_gpus < required_gpus:
            logger.warning(f"⚠️  Sandbox requires {required_gpus} GPU(s), but only {available_gpus} available")
            logger.warning(f"   Local test will use {available_gpus} GPU(s)")
        else:
            logger.info(f"✅ GPU check: {available_gpus} available (sandbox requires {required_gpus})")
        
        # Use first GPU (same as sandbox assigns GPU 0 inside container)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        logger.info("Loading model (same setup as validator)...")
        
        # MUST match validator's runner.py exactly
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},  # Force to specific device (same as validator)
            trust_remote_code=True,
        )
        model.train()  # Note: validator does NOT enable gradient_checkpointing
        
        logger.info("Loading data...")
        data = torch.load(data_path, weights_only=True)
        # Truncate to sequence_length (same as validator)
        data = data[:, :sequence_length]
        
        # Create iterator (same as validator)
        def create_iterator():
            idx = 0
            while True:
                end_idx = idx + batch_size
                if end_idx > data.shape[0]:
                    idx = 0
                    end_idx = batch_size
                yield data[idx:end_idx]
                idx = end_idx
        
        # Create optimizer (MUST match validator exactly)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
        )
        
        # Run inner_steps (exactly as validator would call it)
        logger.info(f"Running inner_steps with {num_steps} steps...")
        logger.info(f"   Batch size: {batch_size}, Sequence length: {sequence_length}")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        result = inner_steps(
            model=model,
            data_iterator=create_iterator(),
            optimizer=optimizer,
            num_steps=num_steps,
            device=device,
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        wall_time = time.perf_counter() - start_time
        
        # Verify result has required fields
        if not hasattr(result, "final_logits"):
            return False, "Result missing 'final_logits' field", 0.0
        if not hasattr(result, "total_tokens"):
            return False, "Result missing 'total_tokens' field", 0.0
        if not hasattr(result, "final_loss"):
            return False, "Result missing 'final_loss' field", 0.0
        
        tps = result.total_tokens / wall_time
        
        return True, f"TPS: {tps:,.0f}, Tokens: {result.total_tokens:,}, Loss: {result.final_loss:.4f}", tps
        
    except Exception as e:
        return False, f"Error running inner_steps: {e}", 0.0
    finally:
        # Cleanup
        if "train_module" in sys.modules:
            del sys.modules["train_module"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Test your training code locally before submitting"
    )
    parser.add_argument(
        "code_path",
        type=Path,
        help="Path to your train.py file"
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip running inner_steps (only validate structure)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of steps to run (default: 2)"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("TEMPLAR TOURNAMENT - LOCAL TEST")
    logger.info("="*60)
    logger.info(f"Testing: {args.code_path}")
    logger.info("")
    
    # Step 1: Validate code structure
    logger.info("Step 1: Checking code structure...")
    is_valid, errors = validate_code_structure(args.code_path)
    
    if not is_valid:
        logger.error("❌ STRUCTURE VALIDATION FAILED:")
        for error in errors:
            logger.error(f"   - {error}")
        logger.info("")
        logger.info("💡 Fix these errors before submitting")
        return 1
    
    logger.info("✅ Code structure valid")
    logger.info("   - inner_steps function found")
    logger.info("   - InnerStepsResult class found")
    logger.info("   - No forbidden imports")
    logger.info("")
    
    # Step 2: Actually run inner_steps
    if not args.skip_run:
        logger.info(f"Step 2: Running inner_steps ({args.steps} steps)...")
        success, message, tps = run_inner_steps(args.code_path, num_steps=args.steps)
        
        if not success:
            logger.error(f"❌ EXECUTION FAILED: {message}")
            return 1
        
        logger.info(f"✅ Execution successful: {message}")
        logger.info("")
    else:
        logger.info("Step 2: Skipped (--skip-run)")
        logger.info("")
    
    logger.info("="*60)
    logger.info("✅ ALL TESTS PASSED - Ready to submit!")
    logger.info("="*60)
    logger.info("")
    logger.info("💡 To submit (costs 0.1 TAO):")
    logger.info("   uv run python -m neurons.miner train.py \\")
    logger.info("       --wallet.name mywallet \\")
    logger.info("       --wallet.hotkey myhotkey \\")
    logger.info("       --payment-recipient <validator_hotkey> \\")
    logger.info("       --validator-api <validator_url>")
    logger.info("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

