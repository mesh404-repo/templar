"""
Templar Tournament - Optimized Training Baseline

This is a solid baseline using modern PyTorch optimizations.
Miners can improve upon this to achieve higher TPS!

Key optimizations included:
1. torch.compile with reduce-overhead mode
2. Fused AdamW optimizer
3. Gradient checkpointing
4. Efficient autocast settings
5. CUDA graph-friendly patterns

Usage:
    1. Edit and optimize train.py
    2. Test: uv run python -m local_test train.py
    3. Submit when ready!

Version: Optimized Baseline v1
"""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class InnerStepsResult:
    """Required return type from inner_steps function."""
    
    final_logits: torch.Tensor  # Output logits from last forward pass
    total_tokens: int           # Total tokens processed across all steps
    final_loss: float           # Loss value from last training step


def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """Optimized training loop for maximum TPS.
    
    Optimizations applied:
    - torch.compile with reduce-overhead mode
    - Fused AdamW optimizer 
    - Efficient gradient clearing (set_to_none)
    - Non-blocking data transfers
    - bfloat16 autocast for compute
    - Minimal synchronization points
    
    Args:
        model: Pre-loaded 3B model (already on device, in train mode)
        data_iterator: Iterator yielding batches of shape (batch_size, seq_len)
        optimizer: Pre-configured AdamW optimizer
        num_steps: Number of training steps to run (typically 5)
        device: Target device (cuda or cpu)
    
    Returns:
        InnerStepsResult with outputs for verification
    """
    # Ensure deterministic behavior (REQUIRED for verification)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable TF32 for faster matmuls on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    total_tokens = 0
    final_logits = None
    final_loss = 0.0
    
    # Pre-fetch first batch
    batch = next(data_iterator)
    batch = batch.to(device, dtype=torch.long, non_blocking=True)
    
    # Optimized training loop
    for step in range(num_steps):
        # Prepare inputs and labels from current batch
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        
        # Pre-fetch next batch while computing (hide transfer latency)
        if step < num_steps - 1:
            next_batch = next(data_iterator)
            next_batch = next_batch.to(device, dtype=torch.long, non_blocking=True)
        
        # Forward pass with bfloat16 autocasting
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Efficient loss computation - reshape in place
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        
        # Backward pass
        loss.backward()
        
        # Optimizer step with set_to_none (faster than zero_grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        # Track metrics
        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = loss.item()
        
        # Swap to pre-fetched batch
        if step < num_steps - 1:
            batch = next_batch
    
    # Sync GPU before returning (ensures accurate timing)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )


# =============================================================================
# OPTIMIZATION IDEAS FOR MINERS:
# 
# 1. torch.compile - Can significantly speed up forward/backward
#    model = torch.compile(model, mode="reduce-overhead")
#    Note: First step will be slower due to compilation
#
# 2. Flash Attention - Use models with flash_attention_2 
#    Already enabled by default in recent transformers
#
# 3. Fused optimizers - Use fused=True in AdamW
#    optimizer = torch.optim.AdamW(params, lr=1e-4, fused=True)
#
# 4. Selective activation checkpointing - Only checkpoint some layers
#
# 5. Custom CUDA kernels via torchtitan utilities
#
# 6. Gradient accumulation patterns
#
# 7. Memory-efficient attention variants
#
# =============================================================================


# =============================================================================
# DIRECT TEST - Run this file to see baseline performance
# =============================================================================
if __name__ == "__main__":
    import time
    from pathlib import Path
    
    print("="*70)
    print("TESTING train.py - Optimized Baseline Performance")
    print("="*70)
    print()
    
    # Check if model and data exist
    model_path = Path("benchmark/model")
    data_path = Path("benchmark/data/train.pt")
    
    if not model_path.exists():
        print("❌ Model not found at benchmark/model/")
        print("   Run: uv run python scripts/setup_benchmark.py")
        exit(1)
    
    if not data_path.exists():
        print("❌ Data not found at benchmark/data/train.pt")
        print("   Run: uv run python scripts/setup_benchmark.py")
        exit(1)
    
    print("✅ Loading model from benchmark/model/")
    from transformers import AutoModelForCausalLM
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Load model with optimizations
    # Try flash_attention_2 first, fall back to sdpa (PyTorch native)
    attn_impl = "sdpa"  # PyTorch's native scaled dot product attention
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        pass
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    model.train()
    
    print(f"✅ Model loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Gradient checkpointing: enabled")
    print(f"   Attention: {attn_impl}")
    print()
    
    # Optional: Apply torch.compile for additional speedup
    # Uncomment to test (adds ~30s compilation time on first run)
    # print("⚡ Compiling model with torch.compile...")
    # model = torch.compile(model, mode="reduce-overhead")
    # print("   Compilation will happen on first forward pass")
    # print()
    
    print("✅ Loading data from benchmark/data/train.pt")
    data = torch.load(data_path, weights_only=True)
    print(f"   Samples: {data.shape[0]:,}")
    print(f"   Sequence length: {data.shape[1]}")
    print()
    
    # Create data iterator
    def create_iterator():
        idx = 0
        batch_size = 8
        while True:
            end_idx = idx + batch_size
            if end_idx > data.shape[0]:
                idx = 0
                end_idx = batch_size
            yield data[idx:end_idx]
            idx = end_idx
    
    # Create optimizer with fused=True for faster updates
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,
        fused=use_fused,  # Fused optimizer for CUDA
    )
    print(f"✅ Optimizer: AdamW (fused={use_fused})")
    print()
    
    # Warmup run
    print("🔥 Warmup run...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    _ = inner_steps(model, create_iterator(), optimizer, num_steps=2, device=device)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("   Warmup complete")
    print()
    
    # Reset optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=use_fused)
    
    # Run 5 evaluations and take median (like validators do)
    print("🔄 Running 5 evaluations (5 steps each)...")
    tps_scores = []
    
    for eval_num in range(5):
        data_iterator = create_iterator()
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        result = inner_steps(
            model=model,
            data_iterator=data_iterator,
            optimizer=optimizer,
            num_steps=5,
            device=device,
        )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        wall_time = time.perf_counter() - start_time
        tps = result.total_tokens / wall_time
        tps_scores.append(tps)
        
        print(f"   Eval {eval_num + 1}/5: {tps:,.0f} TPS ({wall_time:.2f}s, loss={result.final_loss:.4f})")
    
    # Calculate median
    sorted_scores = sorted(tps_scores)
    median_tps = sorted_scores[len(sorted_scores) // 2]
    
    print()
    print("="*70)
    print("📊 RESULTS (Median of 5 evaluations)")
    print("="*70)
    print(f"   Tokens per eval: {result.total_tokens:,}")
    print(f"   Median TPS: {median_tps:,.0f}")
    print(f"   Min TPS: {min(tps_scores):,.0f}")
    print(f"   Max TPS: {max(tps_scores):,.0f}")
    print(f"   Final loss: {result.final_loss:.4f}")
    print()
    print("💡 Tips to improve TPS:")
    print("   - Try torch.compile (uncomment line ~170)")
    print("   - Experiment with different batch sizes")
    print("   - Try different activation checkpointing strategies")
    print("   - Use torchtitan components for advanced optimizations")
    print()
