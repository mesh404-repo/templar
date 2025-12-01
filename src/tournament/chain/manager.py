"""Bittensor chain manager for subnet interactions."""

import asyncio

import bittensor as bt

from ..config import get_config, get_hparams
from ..core.exceptions import ChainError


class ChainManager:
    """Manages interactions with the Bittensor network."""

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

    async def sync_metagraph(self) -> bt.metagraph:
        """Sync and return the metagraph."""
        loop = asyncio.get_event_loop()
        self._metagraph = await loop.run_in_executor(
            None,
            lambda: bt.metagraph(netuid=self.netuid, network=self.config.subtensor_network),
        )
        return self._metagraph

    @property
    def metagraph(self) -> bt.metagraph:
        """Get the cached metagraph."""
        if self._metagraph is None:
            raise ChainError("Metagraph not synced. Call sync_metagraph() first.")
        return self._metagraph

    def get_uid_for_hotkey(self, hotkey: str) -> int | None:
        """Get the UID for a hotkey, or None if not registered."""
        try:
            idx = self.metagraph.hotkeys.index(hotkey)
            return int(self.metagraph.uids[idx])
        except ValueError:
            return None

    def is_registered(self, hotkey: str) -> bool:
        """Check if a hotkey is registered on the subnet."""
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
        import torch

        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.subtensor.set_weights(
                    wallet=self.wallet,
                    netuid=self.netuid,
                    uids=torch.tensor(uids, dtype=torch.int64),
                    weights=torch.tensor(weights, dtype=torch.float32),
                    wait_for_inclusion=False,
                    wait_for_finalization=False,
                ),
            )
            return (True, "Weights set successfully")
        except Exception as e:
            return (False, str(e))
