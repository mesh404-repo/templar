"""Payment verification for validators."""

import logging

import bittensor as bt

logger = logging.getLogger(__name__)


class PaymentVerifier:
    """Verifies that miners have paid for their submissions.
    
    Validators check payment receipts before evaluating miner code.
    """

    def __init__(
        self,
        recipient_address: str,
        subtensor: bt.subtensor | None = None,
        required_amount_rao: int = 100_000_000,
    ):
        """Initialize payment verifier.
        
        Args:
            recipient_address: Expected payment destination (validator's address)
            subtensor: Subtensor instance for checking transactions
            required_amount_rao: Required payment amount (default: 0.1 TAO)
        """
        self.recipient_address = recipient_address
        self.subtensor = subtensor or bt.subtensor(network="finney")
        self.required_amount_rao = required_amount_rao

    async def verify_payment(
        self,
        block_hash: str,
        extrinsic_index: int,
        miner_coldkey: str,
        expected_amount_rao: int | None = None,
    ) -> tuple[bool, str]:
        """Verify a payment transaction.
        
        Args:
            block_hash: Block hash containing the payment
            extrinsic_index: Index of the extrinsic in the block
            miner_coldkey: Miner's coldkey that made the payment
            expected_amount_rao: Expected payment amount (uses default if None)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if expected_amount_rao is None:
            expected_amount_rao = self.required_amount_rao

        try:
            # Get the block
            block = self.subtensor.substrate.get_block(block_hash=block_hash)
            if block is None:
                return False, f"Block not found: {block_hash}"

            # Get extrinsics from block
            extrinsics = block.get("extrinsics", [])
            if extrinsic_index >= len(extrinsics):
                return False, f"Extrinsic index {extrinsic_index} out of range"

            extrinsic = extrinsics[extrinsic_index]

            # Verify it's a balance transfer
            if extrinsic.value.get("call", {}).get("call_module") != "Balances":
                return False, "Not a Balances call"

            call_function = extrinsic.value.get("call", {}).get("call_function")
            if call_function not in ["transfer", "transfer_keep_alive"]:
                return False, f"Not a transfer call: {call_function}"

            # Verify sender
            sender = extrinsic.value.get("address")
            if sender != miner_coldkey:
                return False, f"Sender mismatch: expected {miner_coldkey}, got {sender}"

            # Verify recipient
            call_args = extrinsic.value.get("call", {}).get("call_args", [])
            dest = None
            value = None

            for arg in call_args:
                if arg.get("name") == "dest":
                    dest = arg.get("value")
                elif arg.get("name") == "value":
                    value = arg.get("value")

            if dest != self.recipient_address:
                return (
                    False,
                    f"Recipient mismatch: expected {self.recipient_address}, got {dest}",
                )

            if value is None or value < expected_amount_rao:
                return (
                    False,
                    f"Insufficient payment: expected {expected_amount_rao}, got {value}",
                )

            logger.info(
                f"Payment verified: {value:,} RAO from {sender} "
                f"at block {block_hash}"
            )
            return True, "Payment verified"

        except Exception as e:
            logger.error(f"Error verifying payment: {e}")
            return False, f"Verification error: {e}"

