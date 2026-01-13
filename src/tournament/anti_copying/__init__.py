"""Anti-copying protection module.

This module provides tools to detect and prevent code copying in the tournament.

Key components:
- fingerprint: Structural fingerprinting for cross-validator detection
- similarity: Detailed similarity analysis when both codes are available
- detector: Cross-validator copy detection using blockchain data
"""

from .fingerprint import (
    CodeFingerprint,
    compute_fingerprint,
    compare_fingerprints,
    fingerprints_match,
)
from .similarity import CodeSimilarity, calculate_similarity
from .detector import CrossValidatorDetector, CopyDetectionResult, get_detector

__all__ = [
    # Fingerprinting
    "CodeFingerprint",
    "compute_fingerprint", 
    "compare_fingerprints",
    "fingerprints_match",
    # Similarity
    "CodeSimilarity",
    "calculate_similarity",
    # Cross-validator detection
    "CrossValidatorDetector",
    "CopyDetectionResult",
    "get_detector",
]

