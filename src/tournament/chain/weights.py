"""Weight setting logic - winner-takes-all model with burn fallback."""

import logging

from ..config import get_config
from ..storage.database import Database
from .manager import ChainManager

logger = logging.getLogger(__name__)


class WeightSetter:
    """Handles setting weights on the Bittensor network.

    Implements winner-takes-all: 100% of emissions go to the
    single top-scoring submission.

    If no valid winner exists or burn mode is enabled,
    emissions go to the burn/owner hotkey.
    """

    def __init__(
        self,
        chain: ChainManager,
        database: Database,
        burn_hotkey: str | None = None,
        burn_enabled: bool = False,
    ):
        self.chain = chain
        self.db = database
        self.config = get_config()

        # Burn hotkey - fallback destination for emissions
        self.burn_hotkey = burn_hotkey
        self.burn_enabled = burn_enabled

    async def set_weights(self) -> tuple[bool, str]:
        """Set weights based on current leaderboard.

        Priority:
        1. If burn_enabled, all emissions go to burn_hotkey
        2. Otherwise, winner-takes-all to top submission
        3. If no valid winner, fallback to burn_hotkey

        Returns:
            Tuple of (success, message)
        """
        # Sync metagraph to get latest state
        await self.chain.sync_metagraph()

        # Check if burn mode is enabled
        if self.burn_enabled:
            return await self._set_burn_weights("Burn mode enabled")

        # Get top submission
        top_submission = await self.db.get_top_submission()

        if top_submission is None:
            logger.info("No finished submissions - falling back to burn")
            return await self._set_burn_weights("No finished submissions")

        # Verify miner is still registered
        winner_hotkey = top_submission.miner_hotkey
        if not self.chain.is_registered(winner_hotkey):
            logger.warning(f"Winner {winner_hotkey} not registered - falling back to burn")
            return await self._set_burn_weights(f"Winner {winner_hotkey} not registered")

        # Get UID for winner
        winner_uid = self.chain.get_uid_for_hotkey(winner_hotkey)
        if winner_uid is None:
            logger.error(f"Could not get UID for {winner_hotkey} - falling back to burn")
            return await self._set_burn_weights(f"Could not get UID for {winner_hotkey}")

        # Set 100% weight to winner
        logger.info(
            f"Setting weights: UID {winner_uid} ({winner_hotkey}) "
            f"score={top_submission.final_score:.2f} -> weight=1.0"
        )

        success, message = await self.chain.set_weights(
            uids=[winner_uid],
            weights=[1.0],
        )

        if success:
            logger.info(f"Weights set successfully for UID {winner_uid}")
        else:
            logger.error(f"Failed to set weights: {message}")

        return (success, message)

    async def _set_burn_weights(self, reason: str) -> tuple[bool, str]:
        """Set weights to burn hotkey.

        Args:
            reason: Why we're burning (for logging)

        Returns:
            Tuple of (success, message)
        """
        if self.burn_hotkey is None:
            logger.warning(f"No burn hotkey configured - cannot burn ({reason})")
            return (False, f"No burn hotkey configured: {reason}")

        if not self.chain.is_registered(self.burn_hotkey):
            logger.warning(f"Burn hotkey {self.burn_hotkey} not registered")
            return (False, f"Burn hotkey not registered: {reason}")

        burn_uid = self.chain.get_uid_for_hotkey(self.burn_hotkey)
        if burn_uid is None:
            logger.error(f"Could not get UID for burn hotkey {self.burn_hotkey}")
            return (False, f"Could not get burn UID: {reason}")

        logger.info(f"Setting burn weights: UID {burn_uid} -> weight=1.0 ({reason})")

        success, message = await self.chain.set_weights(
            uids=[burn_uid],
            weights=[1.0],
        )

        if success:
            logger.info(f"Burn weights set successfully for UID {burn_uid}")
        else:
            logger.error(f"Failed to set burn weights: {message}")

        return (success, f"Burn: {message}" if success else message)
