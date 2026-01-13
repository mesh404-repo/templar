# ⚡ Templar Tournament

Compete to write the fastest PyTorch training code. Winner-takes-all: Highest TPS → 100% of subnet emissions.

**Live Dashboard:** Check with your validator for their dashboard URL

---

## 🎯 Overview

**The Challenge:** Optimize training to achieve maximum tokens-per-second (TPS)

- **Task**: Run training steps on Qwen2.5-3B
- **Metric**: TPS = total_tokens / wall_time
- **Reward**: #1 on leaderboard gets 100% emissions
- **Cost**: 0.1 TAO per submission
- **Evaluation**: 5 runs with different seeds, median score used (fair!)

---

## 🚀 For Miners

### Setup
```bash
# 1. Clone and install
git clone https://github.com/tplr-ai/templar-tournament.git
cd templar-tournament
uv sync

# 2. Download model and data
uv run python scripts/setup_miner.py
```

### Test Locally
```bash
# Test your code before submitting (no cost!)
uv run python -m local_test train.py --steps 5

# This validates:
# - Code structure (inner_steps function)
# - Forbidden imports
# - Actual execution with TPS measurement
```

### Submit to Validator
```bash
# One-time: Register on subnet
btcli subnet register --netuid <NETUID> --wallet.name mywallet --wallet.hotkey myhotkey

# Submit code (costs 0.1 TAO)
uv run python -m neurons.miner train.py \
    --wallet.name mywallet \
    --wallet.hotkey myhotkey \
    --payment-recipient <VALIDATOR_HOTKEY_ADDRESS> \
    --validator-api http://<VALIDATOR_IP>:8000
```

**What happens:**
1. Posts code hash + structural fingerprint to blockchain (timestamp proof)
2. Pays 0.1 TAO anti-spam fee
3. Uploads code to validator with cryptographic signature
4. Gets evaluated 5 times (different seeds)
5. Median TPS appears on leaderboard

**Security:** Your code is protected by blockchain timestamp. Even malicious validators can't steal it - you have cryptographic proof you created it first!

### Check Results

Visit the validator's dashboard or use API:
```bash
curl http://<VALIDATOR_IP>:8000/leaderboard
curl http://<VALIDATOR_IP>:8000/api/submissions/<SUBMISSION_ID>
curl http://<VALIDATOR_IP>:8000/api/submissions/<SUBMISSION_ID>/evaluations
```

---

## 🛠️ For Validators

### Setup

```bash
# 1. Install
git clone https://github.com/tplr-ai/templar-tournament.git
cd templar-tournament
uv sync

# 2. Download model and test data
uv run python scripts/setup_validator.py

# 3. Configure environment
cat > .env << 'EOF'
# Wallet
TOURNAMENT_WALLET_NAME=validator
TOURNAMENT_WALLET_HOTKEY=validator_hotkey
TOURNAMENT_SUBTENSOR_NETWORK=finney

# R2 Storage (Cloudflare R2 or S3-compatible)
TOURNAMENT_R2_ACCOUNT_ID=your_account_id
TOURNAMENT_R2_BUCKET_NAME=tournament-submissions
TOURNAMENT_R2_ACCESS_KEY_ID=your_access_key
TOURNAMENT_R2_SECRET_ACCESS_KEY=your_secret_key
EOF

# 4. Build Docker sandbox (includes PyTorch, flash-attn, torchtitan)
cd src/tournament/sandbox
docker build -t tournament-sandbox:latest .
cd ../../..

# 5. Register on blockchain
btcli subnet register --netuid <NETUID> --wallet.name validator --wallet.hotkey validator_hotkey
btcli stake add --amount 1000 --wallet.name validator --wallet.hotkey validator_hotkey
```

### Configure (hparams/hparams.json)

```json
{
    "netuid": 2,
    "evaluation_runs": 5,
    "eval_steps": 5,
    "eval_timeout": 600,
    "benchmark_model_name": "Qwen/Qwen2.5-3B",
    "benchmark_batch_size": 8,
    "benchmark_sequence_length": 1024,
    "submission_cost_rao": 100000000,
    "verification": {
        "output_vector_tolerance": 0.02
    },
    "anti_copying": {
        "submission_cooldown_minutes": 5
    }
}
```

### Run Validator

**Terminal 1 - API Server:**
```bash
uv run python -m api.app
# Dashboard: http://your-ip:8000/
```

**Terminal 2 - Validator:**
```bash
uv run python -m neurons.validator \
    --wallet.name validator \
    --wallet.hotkey validator_hotkey
```

**Announce to miners:**
- API URL: `http://your-ip:8000`
- Payment address: Your validator hotkey SS58 address

---

## 📋 Required Code Format

Your `train.py` must have an `inner_steps` function:

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class InnerStepsResult:
    final_logits: torch.Tensor  # (batch, seq_len-1, vocab_size)
    total_tokens: int           # batch_size * seq_length * num_steps
    final_loss: float

def inner_steps(model, data_iterator, optimizer, num_steps, device):
    """Your optimized training loop.
    
    Args:
        model: HuggingFace model (Qwen2.5-3B) - DO NOT modify architecture
        data_iterator: Yields batches of shape (8, 1024)
        optimizer: AdamW optimizer
        num_steps: Number of training steps
        device: torch.device (cuda)
    
    Returns:
        InnerStepsResult with final_logits, total_tokens, final_loss
    """
    total_tokens = 0
    
    for step in range(num_steps):
        batch = next(data_iterator).to(device)
        
        # Standard causal LM training
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids)
            logits = outputs.logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_tokens += batch.numel()
    
    return InnerStepsResult(
        final_logits=logits.float(),
        total_tokens=total_tokens,
        final_loss=loss.item()
    )
```

**Requirements:**
- ✅ Process correct number of tokens (batch_size × seq_length × num_steps)
- ✅ Output logits within 2% of reference (aggregate difference)
- ✅ Complete within timeout (default: 10 minutes)
- ✅ Use only allowed imports

**Forbidden:**
- ❌ Network access (`socket`, `requests`, `urllib`)
- ❌ Filesystem writes outside /tmp
- ❌ Subprocess/shell execution
- ❌ Model architecture changes

**Available in Sandbox:**
- PyTorch with CUDA 12.8
- Transformers, Accelerate
- flash-attn (flash attention)
- torchtitan (PyTorch native training)

---

## 🏆 How Scoring Works

```
# Each submission is evaluated 5 times with different random seeds
Run 1: seed=12345 → 4,302 TPS
Run 2: seed=67890 → 4,289 TPS  
Run 3: seed=54321 → 4,315 TPS  ← median
Run 4: seed=98765 → 4,298 TPS
Run 5: seed=11111 → 4,305 TPS

Final Score: 4,302 TPS (median protects against GPU hiccups)

# TPS calculation
TPS = total_tokens / wall_time_seconds
```

**Why median?**
- Protects against random GPU slowdowns
- Fair: one bad run doesn't tank your score
- Example: [1200, 1250, 50] → median=1200 (not avg=833)

**Winner gets 100% of emissions!**

Weights update every 10 minutes. Hold #1 → Earn TAO continuously.

---

## 📊 Dashboard Features

Visit validator's dashboard to see:
- 📈 Network stats (submissions, top TPS, active miners)
- ⚡ Validator status (live evaluations)
- 📊 TPS history chart
- 🏆 Leaderboard with rankings
- 💻 Code viewer (submissions visible after evaluation)
- 🔍 Detailed evaluation results per run

---

## 🔒 Security Model

**Multi-Layer Protection:**

| Layer | Protection |
|-------|------------|
| PKE Authentication | Sign submissions with wallet - can't impersonate |
| Blockchain Timestamp | Hash + fingerprint posted to chain FIRST - proves ownership |
| Structural Fingerprint | Detects modified copies across validators |
| Code Hidden | Only visible after evaluation completes |
| Submission Cooldown | 5 minute wait between submissions |
| Isolated Sandbox | Docker with no network, read-only filesystem |

**Anti-Copying Flow:**
1. Miner posts code hash + structural fingerprint to blockchain
2. Block number proves "Miner X had this code at time T"
3. If someone copies: original has earlier block number = wins
4. Structural fingerprint catches modified copies (renamed variables, etc.)

**Verification:**
- Same random seed for reference and miner code
- Model state reset between evaluations
- Output logits must match within 2% aggregate difference

---

## 🔧 Configuration Reference

### hparams.json

| Field | Description | Default |
|-------|-------------|---------|
| `netuid` | Subnet ID | 2 |
| `evaluation_runs` | Runs per submission (median taken) | 5 |
| `eval_steps` | Training steps per run | 5 |
| `eval_timeout` | Max seconds per evaluation | 600 |
| `benchmark_model_name` | HuggingFace model | Qwen/Qwen2.5-3B |
| `benchmark_batch_size` | Batch size | 8 |
| `benchmark_sequence_length` | Sequence length | 1024 |
| `submission_cost_rao` | Cost in RAO (1 TAO = 1e9 RAO) | 100000000 |
| `verification.output_vector_tolerance` | Max aggregate logit diff | 0.02 (2%) |
| `anti_copying.submission_cooldown_minutes` | Wait between submissions | 5 |
| `sandbox.memory_limit` | Container memory | 32g |
| `sandbox.gpu_count` | GPUs per evaluation | 1 |

---

## 🐛 Troubleshooting

**"Output logits don't match"**
- Your code produces different results than reference
- Check: using correct loss function, autocast, optimizer settings

**"Timeout exceeded"**
- Code took too long
- Optimize your training loop

**"Forbidden import"**
- Using disallowed module (os, subprocess, socket, etc.)
- Check imports in your train.py

**"Not registered"**
- Wallet not registered on subnet
- Run: `btcli subnet register --netuid <NETUID> ...`

---

**Ready to compete? Optimize your code and claim #1! ⚡**
