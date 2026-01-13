"""R2/S3 storage for code submissions.

This module handles uploading and downloading miner code submissions to/from
Cloudflare R2 (S3-compatible storage).
"""

import logging
import shutil
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from ..config import get_config

logger = logging.getLogger(__name__)


class R2Storage:
    """Manages code submission storage in Cloudflare R2 (S3-compatible)."""

    def __init__(
        self,
        account_id: str | None = None,
        bucket_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
    ):
        """Initialize R2 storage client.

        Args:
            account_id: R2 account ID (from config if None)
            bucket_name: R2 bucket name (from config if None)
            access_key_id: R2 access key (from config if None)
            secret_access_key: R2 secret key (from config if None)
        """
        config = get_config()

        self.account_id = account_id or config.r2_account_id
        self.bucket_name = bucket_name or config.r2_bucket_name
        self.access_key_id = access_key_id or config.r2_access_key_id
        self.secret_access_key = secret_access_key or config.r2_secret_access_key

        # Check if R2 is configured
        if not all([self.account_id, self.access_key_id, self.secret_access_key]):
            logger.warning(
                "R2 storage not fully configured - will use local filesystem fallback"
            )
            self.client = None
            return

        # Create S3 client configured for R2
        endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"

        try:
            self.client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name="auto",  # R2 uses "auto" region
            )
            logger.info(f"R2 storage initialized: bucket={self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize R2 client: {e}")
            self.client = None

    def is_configured(self) -> bool:
        """Check if R2 storage is properly configured."""
        return self.client is not None

    async def upload_code(self, local_path: str | Path, remote_key: str) -> bool:
        """Upload code file to R2.

        Args:
            local_path: Path to local file to upload
            remote_key: S3 key (path) in the bucket (e.g., "submissions/uid/hash/train.py")

        Returns:
            True if upload succeeded, False otherwise
        """
        if not self.is_configured():
            logger.warning("R2 not configured - storing locally")
            return self._upload_local(local_path, remote_key)

        try:
            local_path = Path(local_path)
            if not local_path.exists():
                logger.error(f"Local file not found: {local_path}")
                return False

            logger.info(f"Uploading {local_path} to R2: {remote_key}")

            with open(local_path, "rb") as f:
                self.client.upload_fileobj(
                    f,
                    self.bucket_name,
                    remote_key,
                    ExtraArgs={"ContentType": "text/x-python"},
                )

            logger.info(f"Upload successful: {remote_key}")
            return True

        except ClientError as e:
            logger.error(f"R2 upload failed: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error during upload: {e}")
            return False

    async def download_code(self, remote_key: str, local_path: str | Path) -> bool:
        """Download code file from R2.

        Args:
            remote_key: S3 key (path) in the bucket
            local_path: Local path to save downloaded file

        Returns:
            True if download succeeded, False otherwise
        """
        if not self.is_configured():
            logger.warning("R2 not configured - loading from local storage")
            return self._download_local(remote_key, local_path)

        try:
            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading {remote_key} from R2 to {local_path}")

            with open(local_path, "wb") as f:
                self.client.download_fileobj(self.bucket_name, remote_key, f)

            logger.info(f"Download successful: {local_path}")
            return True

        except ClientError as e:
            logger.error(f"R2 download failed: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error during download: {e}")
            return False

    async def delete_code(self, remote_key: str) -> bool:
        """Delete code file from R2.

        Args:
            remote_key: S3 key (path) in the bucket

        Returns:
            True if deletion succeeded, False otherwise
        """
        if not self.is_configured():
            logger.warning("R2 not configured - deleting from local storage")
            return self._delete_local(remote_key)

        try:
            logger.info(f"Deleting {remote_key} from R2")
            self.client.delete_object(Bucket=self.bucket_name, Key=remote_key)
            logger.info(f"Deletion successful: {remote_key}")
            return True

        except ClientError as e:
            logger.error(f"R2 deletion failed: {e}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error during deletion: {e}")
            return False

    def _get_local_storage_path(self, remote_key: str) -> Path:
        """Get local filesystem path for storing code when R2 is not configured."""
        return Path("./local_storage") / remote_key

    def _upload_local(self, local_path: str | Path, remote_key: str) -> bool:
        """Fallback: Store file locally when R2 is not configured."""
        try:
            local_path = Path(local_path)
            storage_path = self._get_local_storage_path(remote_key)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, storage_path)
            logger.info(f"Stored locally: {storage_path}")
            return True
        except Exception as e:
            logger.error(f"Local storage failed: {e}")
            return False

    def _download_local(self, remote_key: str, local_path: str | Path) -> bool:
        """Fallback: Load file from local storage when R2 is not configured."""
        try:
            storage_path = self._get_local_storage_path(remote_key)
            if not storage_path.exists():
                logger.error(f"File not found in local storage: {storage_path}")
                return False

            local_path = Path(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(storage_path, local_path)
            logger.info(f"Loaded from local storage: {local_path}")
            return True
        except Exception as e:
            logger.error(f"Local download failed: {e}")
            return False

    def _delete_local(self, remote_key: str) -> bool:
        """Fallback: Delete file from local storage when R2 is not configured."""
        try:
            storage_path = self._get_local_storage_path(remote_key)
            if storage_path.exists():
                storage_path.unlink()
                logger.info(f"Deleted from local storage: {storage_path}")
            return True
        except Exception as e:
            logger.error(f"Local deletion failed: {e}")
            return False


# Global instance
_r2_storage: R2Storage | None = None


def get_r2_storage() -> R2Storage:
    """Get or create global R2Storage instance."""
    global _r2_storage
    if _r2_storage is None:
        _r2_storage = R2Storage()
    return _r2_storage

