#!/usr/bin/env python3
"""
Download model and datasets for VALIDATORS.

This downloads:
- Official 7B model (same as miners)
- Data (seed=12345, PRIVATE, for evaluation)

Usage:
    uv run python scripts/setup_validator.py
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
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_revision,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
            trust_remote_code=True,
        )
        
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model downloaded: {num_params:,} parameters")
        logger.info(f"📁 Saved to: {model_dir}")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise


def download_dataset(data_path: Path, config: dict, dataset_type: str, seed: int) -> None:
    """Download and prepare dataset with given seed."""
    dataset_name = config["benchmark_dataset_name"]
    dataset_split = config["benchmark_dataset_split"]
    dataset_size = config.get(f"benchmark_{dataset_type}_size", 100000)
    seq_length = config["benchmark_sequence_length"]
    model_name = config["benchmark_model_name"]
    
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # TEST DATA: Offset 100k to avoid overlap with training data
        # Uses same master seed (42) for shuffle, but takes samples 100k-200k
        logger.info("⏳ Loading and tokenizing...")
        master_seed = 42  # Same master seed as training (ensures deterministic, non-overlapping split)
        test_offset = 100000  # Start after training data ends
        
        dataset = load_dataset(dataset_name, split=dataset_split, streaming=True)
        dataset = dataset.shuffle(seed=master_seed, buffer_size=10000)  # Same shuffle as train!
        
        tokenized_samples = []
        for i, item in enumerate(dataset):
            # Skip to offset (skip training data samples 0-99,999)
            if i < test_offset:
                continue
            if i >= test_offset + dataset_size:
                break
            
            text = item.get("text", item.get("content", ""))
            if not text:
                continue
            
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
        
        data_tensor = torch.stack(tokenized_samples)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_tensor, data_path)
        
        logger.info(f"✅ Data prepared: {len(tokenized_samples)} samples")
        logger.info(f"📁 Saved to: {data_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        raise


def main():
    logger.info("="*70)
    logger.info("")
    logger.info("VALIDATOR ONLY")
    logger.info("   This includes private test data for evaluation")
    logger.info("")
    logger.info("Downloads:")
    logger.info("  1. Model (same as miners)")
    logger.info("  2. Test data (PRIVATE for evaluation)")
    logger.info("")
    
    config = load_config()
    master_seed = config.get("benchmark_master_seed", 42)
    
    input("Press ENTER to start...")
    logger.info("")
    
    # Download model
    model_dir = Path("benchmark/model")
    download_model(model_dir, config)
    logger.info("")

    test_path = Path("benchmark/data/test.pt")
    download_dataset(test_path, config, "test", master_seed)
    logger.info("")


if __name__ == "__main__":
    main()

