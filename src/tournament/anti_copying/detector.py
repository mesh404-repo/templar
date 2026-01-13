"""Cross-validator copy detection using blockchain data.

This module allows validators to detect copies even if they never
received the original submission, by querying blockchain fingerprints.
"""

import logging
from dataclasses import dataclass

import bittensor as bt

from tournament.config import get_config, get_hparams

logger = logging.getLogger(__name__)


@dataclass 
class CopyDetectionResult:
    """Result of copy detection check."""
    is_potential_copy: bool
    original_hotkey: str | None
    original_block: int | None
    similarity_reason: str
    confidence: str  # "high", "medium", "low"


class CrossValidatorDetector:
    """Detect copies across validators using blockchain data.
    
    Since all submissions post their fingerprint to blockchain,
    any validator can detect if a new submission is similar to
    an existing one, even if they never saw the original.
    """
    
    def __init__(self, subtensor: bt.subtensor | None = None):
        config = get_config()
        self.subtensor = subtensor or bt.subtensor(network=config.subtensor_network)
        self.hparams = get_hparams()
        self.netuid = self.hparams.netuid
        
        # Cache of known fingerprints: {fingerprint: (hotkey, block)}
        self._fingerprint_cache: dict[str, tuple[str, int]] = {}
    
    async def refresh_fingerprint_cache(self) -> None:
        """Refresh cache of known fingerprints from blockchain.
        
        This queries the blockchain for all commitments and builds
        a cache of fingerprints for quick lookup.
        """
        try:
            # Get all commitments from the subnet
            # Note: This is a simplified version - in production you'd
            # query historical commitments more efficiently
            
            metagraph = self.subtensor.metagraph(self.netuid)
            
            for uid, hotkey in enumerate(metagraph.hotkeys):
                try:
                    # Get latest commitment for this hotkey
                    commitment = self.subtensor.get_commitment(
                        netuid=self.netuid,
                        uid=uid,
                    )
                    
                    if commitment:
                        # Parse fingerprint from commitment
                        # Format: "code_hash:fingerprint" or just fingerprint
                        if ":" in commitment:
                            parts = commitment.split(":")
                            fingerprint = parts[1] if len(parts) > 1 else parts[0]
                        else:
                            fingerprint = commitment
                        
                        # Get block number (would need to query commitment history)
                        # For now, use current block as approximation
                        block = self.subtensor.block
                        
                        self._fingerprint_cache[fingerprint] = (hotkey, block)
                        
                except Exception as e:
                    logger.debug(f"Could not get commitment for UID {uid}: {e}")
                    continue
                    
            logger.info(f"Refreshed fingerprint cache: {len(self._fingerprint_cache)} entries")
            
        except Exception as e:
            logger.error(f"Failed to refresh fingerprint cache: {e}")
    
    def check_for_copy(
        self,
        fingerprint: str,
        submitter_hotkey: str,
        submission_block: int,
    ) -> CopyDetectionResult:
        """Check if a submission appears to be a copy based on fingerprint.
        
        Args:
            fingerprint: Structural fingerprint of the new submission
            submitter_hotkey: Hotkey of the submitter
            submission_block: Block number of the submission
            
        Returns:
            CopyDetectionResult indicating if this is a potential copy
        """
        # Check exact fingerprint match
        if fingerprint in self._fingerprint_cache:
            original_hotkey, original_block = self._fingerprint_cache[fingerprint]
            
            if original_hotkey != submitter_hotkey:
                # Different hotkey, same fingerprint = likely copy
                if original_block < submission_block:
                    return CopyDetectionResult(
                        is_potential_copy=True,
                        original_hotkey=original_hotkey,
                        original_block=original_block,
                        similarity_reason="Exact fingerprint match with earlier submission",
                        confidence="high",
                    )
                else:
                    # Our submission was earlier - we're the original
                    return CopyDetectionResult(
                        is_potential_copy=False,
                        original_hotkey=None,
                        original_block=None,
                        similarity_reason="We have earlier timestamp",
                        confidence="high",
                    )
        
        # Check partial fingerprint match (prefix comparison)
        fingerprint_prefix = fingerprint[:16]  # First 16 chars
        
        for cached_fp, (cached_hotkey, cached_block) in self._fingerprint_cache.items():
            if cached_hotkey == submitter_hotkey:
                continue  # Skip own submissions
                
            if cached_fp.startswith(fingerprint_prefix):
                # Partial match - potentially similar code
                if cached_block < submission_block:
                    return CopyDetectionResult(
                        is_potential_copy=True,
                        original_hotkey=cached_hotkey,
                        original_block=cached_block,
                        similarity_reason="Similar fingerprint prefix with earlier submission",
                        confidence="medium",
                    )
        
        # No match found
        return CopyDetectionResult(
            is_potential_copy=False,
            original_hotkey=None,
            original_block=None,
            similarity_reason="No similar fingerprints found",
            confidence="high",
        )
    
    def register_fingerprint(
        self,
        fingerprint: str,
        hotkey: str,
        block: int,
    ) -> None:
        """Register a new fingerprint in the local cache.
        
        Called when we receive a new submission.
        """
        self._fingerprint_cache[fingerprint] = (hotkey, block)
        logger.debug(f"Registered fingerprint {fingerprint[:16]}... for {hotkey[:16]}... at block {block}")


# Global detector instance
_detector: CrossValidatorDetector | None = None


def get_detector() -> CrossValidatorDetector:
    """Get or create the global detector instance."""
    global _detector
    if _detector is None:
        _detector = CrossValidatorDetector()
    return _detector

