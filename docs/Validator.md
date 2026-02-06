# Validator Guide

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              YOUR VALIDATOR MACHINE                              │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         validator.py main loop                          │   │
│   │                                                                         │   │
│   │   1. Read miner commitments from Bittensor blockchain                   │   │
│   │   2. Decrypt timelock-encrypted code URLs                               │   │
│   │   3. Download miner's train.py from URL                                 │   │
│   │   4. Evaluate via Docker or Basilica (multiple runs)                    │   │
│   │   5. Verify: gradients, loss, parameters changed                        │   │
│   │   6. Calculate MFU (Model FLOPs Utilization)                            │   │
│   │   7. Update leaderboard with adaptive threshold                         │   │
│   │   8. Set weights on blockchain (winner gets emissions)                  │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                            │
│                           ┌─────────┴─────────┐                                  │
│                           │  --affinetes-mode │                                  │
│                           └─────────┬─────────┘                                  │
│                                     │                                            │
│               ┌─────────────────────┴─────────────────────┐                      │
│               │                                           │                      │
│               ▼                                           ▼                      │
│      ┌────────────────┐                         ┌────────────────┐               │
│      │  DOCKER MODE   │                         │ BASILICA MODE  │               │
│      │  (Local GPU)   │                         │ (Remote A100)  │               │
│      └───────┬────────┘                         └───────┬────────┘               │
│              │                                          │                        │
└──────────────┼──────────────────────────────────────────┼────────────────────────┘
               │                                          │
               ▼                                          ▼
      ┌─────────────────────┐               ┌──────────────────────────────┐
      │   LOCAL DOCKER      │               │      BASILICA CLOUD          │
      │                     │               │                              │
      │  • Uses YOUR GPU    │               │  • Rents remote A100 GPU     │
      │  • Spawns container │               │  • Pulls your Docker image   │
      │  • templar-eval     │               │  • Runs FastAPI server       │
      │    image            │               │  • POST /evaluate            │
      │  • Full isolation   │               │  • Auto-terminates (TTL)     │
      └─────────────────────┘               └──────────────────────────────┘
```

---

## Key Concepts

### MFU (Model FLOPs Utilization)

The primary metric for ranking miners:

```
MFU = (actual_tflops / gpu_peak_tflops) * 100

Where:
- actual_tflops = model_flops * total_tokens / wall_time / 1e12
- model_flops = 6 * model_params * seq_len (approximate)
- gpu_peak_tflops = 312.0 for A100 (from hparams)
```

Higher MFU = more efficient use of GPU compute.

### Verification Checks

Each submission is verified to prevent cheating:

| Check | Description | Threshold |
|-------|-------------|-----------|
| **Loss Validity** | Loss must be valid (not NaN, not too high) | `max_loss_difference: 0.3` |
| **Gradient Cosine** | Gradients must align with reference | `gradient_cosine_min: 0.95` |
| **Gradient Norm** | Gradient magnitudes must be reasonable | `0.9 - 1.1` ratio |
| **Trainable Params** | All params must be trainable | `100%` |
| **Params Changed** | Most params must change during training | `min: 80%` |
| **Success Rate** | Majority of runs must pass | `min_success_rate: 0.5` |

### Adaptive Threshold & Leaderboard

The leaderboard uses an adaptive threshold to prevent marginal improvements from stealing the crown:

```
MFU to Beat = Current Leader MFU × (1 + threshold)

Example:
- Leader: 45% MFU
- Threshold: 10%
- MFU to Beat: 45% × 1.10 = 49.5%
```

**Threshold behavior:**
- **Increases** when a new leader makes a big improvement (e.g., 45% → 60% = 33% improvement → threshold becomes 33%)
- **Decays** over time towards base (1%) to allow catching up

### Weight Distribution

```
Winner (Rank #1):  5% of emissions
Validator (Burn):  95% of emissions
```

Only the threshold-adjusted rank #1 receives emissions. All others get nothing.

---

## Quick Start

### Docker Mode (Local GPU)

```bash
# 1. Build evaluation image (from repo root)
docker build --network=host -f environments/templar/Dockerfile --no-cache -t templar-eval:latest .

# 2. Run validator
SUBTENSOR_NETWORK=finney \
ENABLE_LOKI=true \
PYTHONUNBUFFERED=1 \
uv run -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --affinetes-mode docker \
    2>&1 | tee logs/validator.log
```

### Basilica Mode (Remote GPU)

```bash
# 1. Build and push image to registry (from repo root)
docker build --network=host -f environments/templar/Dockerfile -t ghcr.io/one-covenant/templar-eval:latest .
docker push ghcr.io/one-covenant/templar-eval:latest

# 2. Run validator (no local GPU needed!)
SUBTENSOR_NETWORK=finney \
BASILICA_API_TOKEN="your-token" \
ENABLE_LOKI=true \
PYTHONUNBUFFERED=1 \
uv run -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --affinetes-mode basilica \
    2>&1 | tee logs/validator.log
```

---

## Docker Mode (Detailed)

### Prerequisites

1. **NVIDIA GPU** with CUDA support (A100 recommended)
2. **Docker** with GPU support (`nvidia-container-toolkit`)
3. **Built evaluation image**

### Step 1: Verify GPU Access

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Step 2: Build Evaluation Image

```bash
# Run from repo root - reads model/dataset from hparams.json
docker build --network=host -f environments/templar/Dockerfile --no-cache -t templar-eval:latest .
```

### Step 3: Configure hparams.json

Edit `hparams/hparams.json`:

```json
{
    "netuid": 3,
    "burn_rate": 0.95,
    "burn_uid": 1,
    
    "evaluation_runs": 5,
    "eval_steps": 5,
    "min_success_rate": 0.5,
    
    "docker": {
        "gpu_devices": "0",
        "memory_limit": "80g",
        "shm_size": "32g"
    },
    
    "verification": {
        "max_loss_difference": 0.3,
        "min_params_changed_ratio": 0.8,
        "gradient_cosine_min": 0.95,
        "gradient_norm_ratio_min": 0.9,
        "gradient_norm_ratio_max": 1.1
    },
    
    "adaptive_threshold": {
        "base_threshold": 0.01,
        "decay_percent": 0.05,
        "decay_interval_blocks": 100
    }
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| `netuid` | Subnet ID for Crusades | `3` |
| `burn_rate` | % of emissions to validator | `0.95` (95%) |
| `evaluation_runs` | Number of evaluation runs per submission | `5` |
| `min_success_rate` | Minimum passing runs to accept | `0.5` (50%) |
| `gpu_devices` | Which GPUs to use | `"0"` |
| `memory_limit` | Container memory limit | `"80g"` |

### Step 4: Run Validator

```bash
# Mainnet (Production)
SUBTENSOR_NETWORK=finney \
ENABLE_LOKI=true \
PYTHONUNBUFFERED=1 \
uv run -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --affinetes-mode docker \
    2>&1 | tee logs/validator.log
```

---

## Basilica Mode (Detailed)

### How Basilica Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BASILICA FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   FIRST EVALUATION (~2-4 minutes)                                           │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │ 1. Validator requests deployment from Basilica API               │      │
│   │ 2. Basilica provisions A100 GPU                                  │      │
│   │ 3. Basilica pulls Docker image from ghcr.io                      │      │
│   │ 4. Basilica starts FastAPI server (uvicorn)                      │      │
│   │ 5. Validator calls POST /evaluate with miner's code              │      │
│   │ 6. Basilica returns MFU results                                  │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   SUBSEQUENT EVALUATIONS (instant)                                          │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │ 1. Validator reuses existing deployment                          │      │
│   │ 2. Validator calls POST /evaluate with miner's code              │      │
│   │ 3. Basilica returns MFU results                                  │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
│   AFTER TTL EXPIRES (default: 1 hour)                                       │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │ 1. Deployment auto-terminates                                    │      │
│   │ 2. GPU released                                                  │      │
│   │ 3. Next evaluation creates new deployment                        │      │
│   └──────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Prerequisites

1. **Basilica API Token** (contact Basilica team)
2. **Docker image pushed to public registry** (ghcr.io)
3. **No local GPU required!**

### Step 1: Get Basilica API Token

Contact the Basilica team to get your `BASILICA_API_TOKEN`.

### Step 2: Build and Push Docker Image

```bash
# Build the evaluation image (from repo root)
docker build --network=host -f environments/templar/Dockerfile -t ghcr.io/one-covenant/templar-eval:latest .

# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Push the image (must be PUBLIC for Basilica to pull)
docker push ghcr.io/one-covenant/templar-eval:latest
```

> ⚠️ **Important**: The image must be **public** in ghcr.io settings for Basilica to pull it.

### Step 3: Configure hparams.json

Edit `hparams/hparams.json`:

```json
{
    "netuid": 3,
    "basilica": {
        "image": "ghcr.io/one-covenant/templar-eval:latest",
        "ttl_seconds": 3600,
        "gpu_count": 1,
        "gpu_models": ["A100"],
        "min_gpu_memory_gb": 64,
        "cpu": "4",
        "memory": "32Gi"
    }
}
```

| Setting | Description | Default |
|---------|-------------|---------|
| `image` | Docker image URL | Required |
| `ttl_seconds` | Deployment lifetime | 3600 (1 hour) |
| `gpu_count` | Number of GPUs | 1 |
| `gpu_models` | Acceptable GPU types | ["A100"] |
| `min_gpu_memory_gb` | Minimum VRAM | 64 |

### Step 4: Run Validator

```bash
SUBTENSOR_NETWORK=finney \
BASILICA_API_TOKEN="basilica_xxxxx..." \
ENABLE_LOKI=true \
PYTHONUNBUFFERED=1 \
uv run -m neurons.validator \
    --wallet.name your_wallet \
    --wallet.hotkey your_hotkey \
    --affinetes-mode basilica \
    2>&1 | tee logs/validator.log
```

---

## Database Structure

The validator stores all data in `crusades.db` (SQLite):

| Table | Purpose |
|-------|---------|
| `submissions` | All miner submissions with MFU scores |
| `evaluations` | Individual evaluation runs per submission |
| `adaptive_threshold` | Current threshold state |
| `validator_state` | Winner tracking, block sync state |

### Key Fields

**submissions:**
- `submission_id` - Unique ID
- `miner_hotkey` - Miner's wallet
- `final_score` - MFU percentage
- `status` - pending/evaluating/finished/failed
- `code_content` - Miner's train.py (stored after eval)

**validator_state:**
- `previous_winner_id` - Current leader's submission ID
- `previous_winner_score` - Current leader's MFU
- `last_processed_block` - Sync checkpoint

---

## Logging & Monitoring

### Loki (Grafana)

Enable remote logging to Grafana:

```bash
ENABLE_LOKI=true uv run -m neurons.validator ...
```

View logs at: https://grafana.tplr.ai/dashboards

Query examples:
```
{service="crusades-validator"}
{service="crusades-validator", host="your-hostname"}
{service="crusades-validator"} |= "PASSED"
```

### TUI Dashboard

Real-time monitoring from local database:

```bash
uv run -m crusades.tui --db crusades.db
```

Features:
- Leaderboard with MFU scores
- Recent submissions and status
- MFU history chart
- Evaluation queue
- Adaptive threshold display

### Log Files

```bash
# Real-time logs
tail -f logs/validator.log

# Search for evaluations
grep -E "MFU|PASSED|FAILED|NEW LEADER" logs/validator.log
```

---

## Admin Scripts

### View Submissions

```bash
# List all submissions
uv run scripts/view_submission.py

# Filter by competition version
uv run scripts/view_submission.py --version 2

# View specific submission details
uv run scripts/view_submission.py v2_commit_12345_1

# Save miner's code to file
uv run scripts/view_submission.py v2_commit_12345_1 --save
```