"""Base node with common functionality for validators."""

import asyncio
import logging
import signal
from abc import ABC, abstractmethod

import bittensor as bt

from tournament.chain.manager import ChainManager
from tournament.config import get_config, get_hparams

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Base class for tournament nodes (validators).

    Provides:
    - Signal handling for graceful shutdown
    - Metagraph sync
    """

    def __init__(
        self,
        wallet: bt.wallet | None = None,
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

        # Initialize chain manager
        self.chain = ChainManager(wallet=self.wallet)

        # State
        self.running: bool = False

        # Events
        self.stop_event = asyncio.Event()

        # Setup signal handlers
        self._setup_signals()

    def _setup_signals(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop_event.set()

    @property
    def hotkey(self) -> str:
        """Get the node's hotkey address."""
        return self.wallet.hotkey.ss58_address

    @property
    def uid(self) -> int | None:
        """Get the node's UID on the subnet."""
        return self.chain.get_uid_for_hotkey(self.hotkey)

    async def sync(self) -> None:
        """Sync metagraph."""
        await self.chain.sync_metagraph()
        logger.debug("Metagraph synced")

    @abstractmethod
    async def run_step(self) -> None:
        """Run one iteration of the node's main loop."""
        pass

    async def start(self) -> None:
        """Start the node."""
        logger.info(f"Starting {self.__class__.__name__}...")
        logger.info(f"Hotkey: {self.hotkey}")

        # Initial sync
        # Skip blockchain checks if in test mode
        skip_check = getattr(self, 'skip_blockchain_check', False)
        
        if not skip_check:
            await self.sync()

            if self.uid is None:
                logger.error(f"Hotkey {self.hotkey} not registered on subnet {self.hparams.netuid}")
                return

            logger.info(f"UID: {self.uid}")
        else:
            logger.warning("Skipping blockchain registration check (test mode)")
            logger.info(f"Hotkey: {self.hotkey}")

        self.running = True

        try:
            while not self.stop_event.is_set():
                await self.run_step()
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
            raise
        finally:
            self.running = False
            await self.cleanup()

    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up...")
