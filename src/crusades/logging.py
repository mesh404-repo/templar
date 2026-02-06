"""Loki logging for Grafana integration.

Sends validator logs to Loki at logs.tplr.ai for visualization in Grafana.
Dashboard: https://grafana.tplr.ai/dashboards

Environment variables:
    ENABLE_LOKI: Set to "true" or "1" to enable Loki logging (default: disabled)
    LOKI_URL: Override the Loki push endpoint (default: https://logs.tplr.ai/loki/api/v1/push)
"""

import logging
import logging.handlers
import os
import socket
import time
import uuid
from queue import Queue
from typing import Final

import logging_loki

# Default Loki URL - same as Templar
LOKI_URL: Final[str] = os.environ.get(
    "LOKI_URL", "https://logs.tplr.ai/loki/api/v1/push"
)
TRACE_ID: Final[str] = str(uuid.uuid4())

# Only enable Loki if explicitly set (prevents local runs from polluting prod logs)
ENABLE_LOKI: Final[bool] = os.environ.get("ENABLE_LOKI", "").lower() in ("true", "1")


def setup_loki_logger(
    service: str,
    uid: str,
    version: str,
    environment: str = "finney",
    url: str | None = None,
) -> logging.Logger:
    """Configure logger to send logs to Loki for Grafana.

    Loki logging is DISABLED by default to prevent local testing from
    polluting production logs. Set ENABLE_LOKI=true to enable.

    Args:
        service: Service name (e.g., 'crusades-validator')
        uid: Validator UID
        version: Code version
        environment: Network environment (finney, testnet, local)
        url: Loki push URL (defaults to LOKI_URL env var)

    Returns:
        Configured logger
    """
    logger = logging.getLogger("crusades")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Console handler (always enabled) - use UTC for consistency with Grafana
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s UTC | %(levelname)s | %(name)s | %(message)s"
    )
    formatter.converter = time.gmtime  # Force UTC
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger (avoids duplicates)
    logger.propagate = False

    # Check if Loki is enabled
    if not ENABLE_LOKI:
        logger.info("Loki logging disabled (set ENABLE_LOKI=true to enable)")
        return logger

    url = url or LOKI_URL
    host = socket.gethostname()
    pid = os.getpid()

    tags = {
        "service": service,
        "host": host,
        "pid": str(pid),
        "environment": environment,
        "version": version,
        "uid": uid,
        "trace_id": TRACE_ID,
    }

    # Use queue-based handler for async logging (non-blocking)
    log_queue: Queue = Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    listener = logging.handlers.QueueListener(
        log_queue, respect_handler_level=True
    )

    # Loki handler
    loki_handler = logging_loki.LokiHandler(
        url=url,
        tags=tags,
        version="1",
    )
    loki_handler.setLevel(logging.INFO)

    listener.handlers = (loki_handler,)
    listener.start()

    logger.addHandler(queue_handler)

    logger.info(f"Loki logging enabled: {url}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Trace ID: {TRACE_ID}")

    return logger
