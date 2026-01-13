"""Bittensor chain manager for subnet interactions.

BLOCKCHAIN INTEGRATION:
- Connects to Bittensor network (mainnet or testnet)
- Syncs metagraph (gets all registered miners)
- Verifies miner registration
- Sets weights (distributes incentives)
- Reads current block

All on-chain operations happen through this manager.
"""

import asyncio
import logging

import bittensor as bt
import torch

from ..config import get_config, get_hparams
from ..core.exceptions import ChainError

logger = logging.getLogger(__name__)


class ChainManager:
    """Manages interactions with the Bittensor blockchain.
    
    Handles:
    - Metagraph syncing (who is registered)
    - Weight setting (incentive distribution)
    - Miner verification (is hotkey registered?)
    - Block queries
    """

    def __init__(
        self,
        wallet: bt.wallet | None = None,
        subtensor: bt.subtensor | None = None,
    ):
        self.config = get_config()
        self.hparams = get_hparams()

        # Initialize wallet
        if wallet is None:
            self.wallet = bt.wallet(
                name=self.config.wallet_name,
                hotkey=self.config.wallet_hotkey,
            )
        else:
            self.wallet = wallet

        # Initialize subtensor connection
        if subtensor is None:
            self.subtensor = bt.subtensor(network=self.config.subtensor_network)
        else:
            self.subtensor = subtensor

        self._metagraph: bt.metagraph | None = None

    @property
    def netuid(self) -> int:
        """Get the subnet UID."""
        return self.hparams.netuid

    @property
    def hotkey(self) -> str:
        """Get the wallet's hotkey address."""
        return self.wallet.hotkey.ss58_address

    async def sync_metagraph(self) -> bt.metagraph | None:
        """Sync and return the metagraph.
        
        Returns:
            The synced metagraph, or None if sync fails.
        """
        loop = asyncio.get_event_loop()
        try:
            self._metagraph = await loop.run_in_executor(
                None,
                lambda: bt.metagraph(netuid=self.netuid, network=self.config.subtensor_network),
            )
            logger.info(f"Metagraph synced: {len(self._metagraph.hotkeys)} neurons")
        except Exception as e:
            logger.error(f"Metagraph sync failed: {e}")
            logger.error("Weight setting will be unavailable until metagraph syncs")
            self._metagraph = None
        return self._metagraph

    @property
    def metagraph(self) -> bt.metagraph | None:
        """Get the cached metagraph (None if sync failed)."""
        return self._metagraph

    def get_uid_for_hotkey(self, hotkey: str) -> int | None:
        """Get the UID for a hotkey, or None if not registered."""
        if self.metagraph is None:
            logger.warning("Metagraph not synced - cannot get UID")
            return None
        try:
            idx = self.metagraph.hotkeys.index(hotkey)
            return int(self.metagraph.uids[idx])
        except ValueError:
            return None

    def is_registered(self, hotkey: str) -> bool:
        """Check if a hotkey is registered on the subnet."""
        if self.metagraph is None:
            logger.warning("Metagraph not synced - cannot check registration")
            return False
        return hotkey in self.metagraph.hotkeys

    async def get_current_block(self) -> int:
        """Get the current block number."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.subtensor.block,
        )

    async def set_weights(
        self,
        uids: list[int],
        weights: list[float],
    ) -> tuple[bool, str]:
        """Set weights on the subnet.

        Args:
            uids: List of UIDs to set weights for
            weights: List of weights (should sum to 1.0)

        Returns:
            Tuple of (success, message)
        """
        loop = asyncio.get_event_loop()

        try:
            # Convert to tensors
            uids_tensor = torch.tensor(uids, dtype=torch.int64)
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            
            logger.info(f"Setting weights: UIDs={uids}, weights={weights}")
            
            # Call set_weights and capture result
            success, message = await loop.run_in_executor(
                None,
                lambda: self.subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=self.netuid,
                    uids=uids_tensor,
                    weights=weights_tensor,
                    wait_for_inclusion=True,  # Wait for inclusion
                    wait_for_finalization=False,
                ),
            )
            
            if success:
                logger.info(f"Weights set successfully: {message}")
            else:
                logger.error(f"Weight setting failed: {message}")
            
            return (success, message)
        except Exception as e:
            logger.error(f"Exception during weight setting: {e}")
            return (False, str(e))
