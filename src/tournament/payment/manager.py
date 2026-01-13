"""Payment manager for handling submission payments."""

import logging
import time
from dataclasses import dataclass

import bittensor as bt

logger = logging.getLogger(__name__)


@dataclass
class PaymentReceipt:
    """Receipt from a payment transaction."""

    block_hash: str
    extrinsic_index: int
    amount_rao: int
    timestamp: float


class PaymentManager:
    """Manages payment transactions for submissions.
    
    This implements the Ridges-style pay-to-submit mechanism to prevent spam.
    Miners must pay a small fee (in TAO) before their code is evaluated.
    """

    # Default submission cost (0.1 TAO = 100,000,000 RAO)
    DEFAULT_SUBMISSION_COST_RAO = 100_000_000

    def __init__(
        self,
        wallet: bt.wallet,
        subtensor: bt.subtensor | None = None,
        submission_cost_rao: int | None = None,
    ):
        """Initialize payment manager.
        
        Args:
            wallet: Miner's wallet for making payments
            subtensor: Subtensor instance (created if None)
            submission_cost_rao: Cost per submission in RAO (default: 0.1 TAO)
        """
        self.wallet = wallet
        self.subtensor = subtensor or bt.subtensor(network="finney")
        self.submission_cost_rao = submission_cost_rao or self.DEFAULT_SUBMISSION_COST_RAO

    def get_submission_cost(self) -> tuple[int, float]:
        """Get the cost to submit code.
        
        Returns:
            Tuple of (cost_in_rao, cost_in_tao)
        """
        cost_tao = self.submission_cost_rao / 1e9
        return self.submission_cost_rao, cost_tao

    async def make_payment(
        self,
        recipient_address: str,
        amount_rao: int | None = None,
    ) -> PaymentReceipt:
        """Make a payment for submission.
        
        Args:
            recipient_address: SS58 address to send payment to
            amount_rao: Amount to send in RAO (uses default if None)
            
        Returns:
            PaymentReceipt with transaction details
            
        Raises:
            Exception: If payment fails
        """
        if amount_rao is None:
            amount_rao = self.submission_cost_rao

        logger.info(
            f"Making payment: {amount_rao:,} RAO ({amount_rao/1e9:.4f} TAO) "
            f"to {recipient_address}"
        )

        # Compose the transfer call
        payment_payload = self.subtensor.substrate.compose_call(
            call_module="Balances",
            call_function="transfer_keep_alive",
            call_params={
                "dest": recipient_address,
                "value": amount_rao,
            },
        )

        # Create signed extrinsic
        payment_extrinsic = self.subtensor.substrate.create_signed_extrinsic(
            call=payment_payload,
            keypair=self.wallet.coldkey,
        )

        # Submit and wait for inclusion
        logger.info("Submitting payment transaction...")
        receipt = self.subtensor.substrate.submit_extrinsic(
            payment_extrinsic,
            wait_for_inclusion=True,
        )

        logger.info(
            f"Payment confirmed: block {receipt.block_hash}, "
            f"extrinsic {receipt.extrinsic_idx}"
        )

        return PaymentReceipt(
            block_hash=receipt.block_hash,
            extrinsic_index=receipt.extrinsic_idx,
            amount_rao=amount_rao,
            timestamp=time.time(),
        )

