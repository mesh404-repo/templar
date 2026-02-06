#!/usr/bin/env python3
"""Pre-download dataset for offline evaluation.

This script runs during Docker build to cache dataset samples.
The cached data is shuffled at runtime using validator-provided seed.
"""

import json
import os
from pathlib import Path

from datasets import load_dataset

# Configuration from environment (set by Dockerfile build args)
DATASET_NAME = os.environ.get("DATASET_NAME", "HuggingFaceFW/fineweb")
_num_samples_str = os.environ.get("NUM_SAMPLES", "50000")
try:
    NUM_SAMPLES = int(_num_samples_str)
    if NUM_SAMPLES <= 0:
        raise ValueError("NUM_SAMPLES must be positive")
except ValueError as e:
    raise ValueError(f"Invalid NUM_SAMPLES='{_num_samples_str}': {e}") from e

CACHE_PATH = Path("/home/appuser/.cache/templar")
OUTPUT_FILE = CACHE_PATH / "dataset.json"


def main():
    print(f"Downloading {NUM_SAMPLES} samples from {DATASET_NAME}...")

    # Create cache directory
    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    # Load dataset with streaming
    dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    # Collect text samples
    samples = []
    for i, sample in enumerate(dataset):
        if i >= NUM_SAMPLES:
            break
        text = sample.get("text", sample.get("content", ""))
        if text and len(text) > 100:  # Skip very short samples
            samples.append(text)
        if (i + 1) % 10000 == 0:
            print(f"  Downloaded {i + 1} samples...")

    print(f"Collected {len(samples)} valid samples")

    # Save to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(samples, f)

    file_size = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved to {OUTPUT_FILE} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
