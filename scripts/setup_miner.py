#!/usr/bin/env python3
"""
Download model and data for MINERS.

Usage:
    uv run python scripts/setup_miner.py
"""

import json
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_config():
    """Load benchmark configuration from hparams.json."""
    config_path = Path(__file__).parent.parent / "hparams" / "hparams.json"
    with open(config_path) as f:
        return json.load(f)


def download_model(model_dir: Path, config: dict) -> None:
    """Download the benchmark model (from hparams.json)."""
    model_name = config["benchmark_model_name"]
    model_revision = config.get("benchmark_model_revision", "main")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Download model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_revision,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
            trust_remote_code=True,
        )
        
        # Save to local directory
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model downloaded: {num_params:,} parameters")
        logger.info(f"📁 Saved to: {model_dir}")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise


def download_train_data(data_path: Path, config: dict) -> None:
    """Download and prepare TRAINING dataset (public, for miners).
    
    Uses master seed for shuffle, takes samples 0-99,999.
    Test data takes samples 100,000-199,999 (zero overlap).
    """
    dataset_name = config["benchmark_dataset_name"]
    dataset_split = config["benchmark_dataset_split"]
    dataset_size = config.get("benchmark_train_size", 100000)
    master_seed = config.get("benchmark_master_seed", 42)
    seq_length = config["benchmark_sequence_length"]
    model_name = config["benchmark_model_name"]
    
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load dataset with deterministic sampling
        train_offset = 0  # Training data starts at 0
        
        dataset = load_dataset(
            dataset_name,
            split=dataset_split,
            streaming=True,
        )
        
        # Shuffle with master seed (same shuffle order for train and test)
        # Train takes first 100k, test takes next 100k = ZERO overlap
        dataset = dataset.shuffle(seed=master_seed, buffer_size=10000)
        tokenized_samples = []
        
        for i, item in enumerate(dataset):
            # Skip to offset
            if i < train_offset:
                continue
            if i >= train_offset + dataset_size:
                break
            
            # Get text field
            text = item.get("text", item.get("content", ""))
            if not text:
                continue
            
            # Tokenize
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=seq_length,
                truncation=True,
                padding="max_length",
            )["input_ids"][0]
            
            tokenized_samples.append(tokens)
            
            # Log progress based on actual samples collected
            if len(tokenized_samples) % 10000 == 0:
                logger.info(f"   Processed {len(tokenized_samples)}/{dataset_size}...")
        
        # Stack into tensor
        data_tensor = torch.stack(tokenized_samples)
        
        # Save
        data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_tensor, data_path)
        
        logger.info(f"✅ Training data prepared: {len(tokenized_samples)} samples")
        logger.info(f"   Total tokens: {data_tensor.numel():,}")
        logger.info(f"📁 Saved to: {data_path}")
        
    except Exception as e:
        logger.error(f"❌ Data preparation failed: {e}")
        raise


def main():
    logger.info("="*70)
    logger.info("TEMPLAR TOURNAMENT - BENCHMARK SETUP")
    logger.info("="*70)
    logger.info("")
    logger.info("This downloads the OFFICIAL 3B model and TRAINING data:")
    logger.info("  ✓ Miners use this for local testing")
    logger.info("")
    
    # Load configuration
    config = load_config()
    
    model_dir = Path("benchmark/model")
    train_data_path = Path("benchmark/data/train.pt")
    
    logger.info("📋 Configuration (from hparams.json):")
    logger.info(f"   Model: {config['benchmark_model_name']}")
    logger.info(f"   Dataset: {config['benchmark_dataset_name']}")
    logger.info(f"   Training samples: {config.get('benchmark_train_size', 100000):,}")
    logger.info("")
    
    input("Press ENTER to start download...")
    logger.info("")
    
    # Download model
    download_model(model_dir, config)
    logger.info("")
    
    # Download training data
    download_train_data(train_data_path, config)
    logger.info("")
    
    logger.info("="*70)
    logger.info("✅ SETUP COMPLETE")
    logger.info("="*70)
    logger.info("")
    logger.info("📁 Files created:")
    logger.info(f"   Model: {model_dir}/")
    logger.info(f"   Training data: {train_data_path}")
    logger.info("")
    logger.info("✅ You have everything needed to compete!")
    logger.info("")
    logger.info("📝 Next steps:")
    logger.info("   1. Edit train.py and optimize the inner_steps function")
    logger.info("   2. Test: uv run python train.py  (see your TPS)")
    logger.info("   3. Validate: uv run python -m local_test train.py")
    logger.info("   4. Submit when ready and compete!")


if __name__ == "__main__":
    main()
