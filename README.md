# Templar Crusades

**TPS Crusades on Bittensor** - Miners compete to optimize training code for maximum MFU.

## How It Works

```
┌───────────────────────────────────────────────────────────────────────────────── ┐
│                              Crusades FLOW                                       │
│                                                                                  │
│   MINER                        BLOCKCHAIN                      VALIDATOR         │
│     │                                                              │             │
│     │  1. Host train.py at URL                                     │             │
│     │     (Gist, Pastebin, etc)                                    │             │
│     │                                                              │             │
│     ├──▶ 2. Submit URL ─────────▶ set_reveal_commitment            │             │
│     │                             (timelock encrypted)             │             │
│     │                                    │                         │             │
│     │                                    │ (wait reveal_blocks)    │             │
│     │                                    ▼                         │             │
│     │                              3. Decrypted ◀───────────────── ┤ Read        │
│     │                                                              │             │
│     │                                                     4. Download code       │
│     │                                                        from URL            │
│     │                                                              │             │
│     │                                                     5. Runs in Container   │
│     │                                                        (X eval runs)       │
│     │                                                              │             │
│     │                                                     6. Calculate MFU       │
│     │                                                        (median score)      │
│     │                                                              │             │
│     │                                                     7. Set weights         │
│                                                                                  │
└───────────────────────────────────────────────────────────────────────────────── ┘
```

## Quick Start

### Prerequisites

```bash
# Clone and setup
git clone https://github.com/one-covenant/crusades
cd crusades
uv sync

# Create .env (for HuggingFace access)
echo "HF_TOKEN=hf_your_token" > .env
```

---

## For Miners

### 1. Setup & Test Locally

```bash
# Download model & data for local testing
uv run local_test/setup_benchmark.py

# Test your train.py locally (performance test)
uv run local_test/train.py

# Verify your submission to avoid potential failures during validator checks
uv run local_test/verify.py
```

### 2. Host Your Code

Host your `train.py` at any URL that returns raw code:
- **GitHub Gist** (recommended - use secret gist for privacy)
- **Raw GitHub file** (use raw.githubusercontent.com)
- **Pastebin** or any paste service
- Any HTTP/HTTPS URL

### 3. Submit to Crusades

```bash

# Submit to mainnet
uv run -m neurons.miner submit "https://gist.github.com/user/gist_id" \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --network finney

# Submit to localnet (testing)
uv run -m neurons.miner submit "https://gist.github.com/user/gist_id" \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --network local
```

**Parameters**: `--wallet.name`, `--wallet.hotkey`, `--network` (finney/test/local)

---

## For Validators

See [docs/Validator.md](docs/Validator.md) for detailed validator setup.

---

## train.py Requirements

Your `train.py` must implement the `inner_steps` function. Here's the basic implementation:

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor  # Output logits from last forward pass
    total_tokens: int           # Total tokens processed across all steps
    final_loss: float           # Loss value from last training step

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
    
    for step in range(num_steps):
        # Get batch
        batch = next(data_iterator)
        batch = batch.to(device)
        
        # Prepare inputs and labels
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        
        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        
        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        # Track metrics
        total_tokens += batch.numel()
        final_logits = logits.detach().float()
        final_loss = loss.item()
    
    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens=total_tokens,
        final_loss=final_loss,
    )
```

This is a basic implementation - miners should optimize it for better performance.

---

## Configuration

Key settings in `hparams/hparams.json`:

| Setting | Default | Description |
|---------|---------|-------------|
| `netuid` | 3 | Subnet ID |
| `evaluation_runs` | 5 | Runs per submission (median taken) |
| `eval_steps` | 5 | Training steps per evaluation |
| `benchmark_model_name` | Qwen/Qwen2.5-3B | Model for evaluation |
| `benchmark_batch_size` | 4 | Batch size for evaluation |

---

## TUI Dashboard

Monitor crusades activity in real-time with the terminal dashboard.

```bash
# Connect to the official Crusades API
uv run -m crusades.tui --url 69.19.137.219:8080
```

### Features

- Leaderboard with MFU scores
- Recent submissions and their status
- MFU history chart
- Validator status
- View submission code (after evaluation)