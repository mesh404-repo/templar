"""Templar Crusades - Training code efficiency crusades subnet."""

__version__ = "0.2.0"

# Competition version from major.minor version number
# Major OR Minor bump = new competition (fresh start)
# Patch bump only = same competition continues
# Examples: "0.2.0" → 2, "0.3.0" → 3, "1.0.0" → 100
_version_parts = __version__.split(".")
COMPETITION_VERSION: int = int(_version_parts[0]) * 100 + int(_version_parts[1])

from crusades.logging import LOKI_URL, setup_loki_logger

__all__ = ["__version__", "COMPETITION_VERSION", "setup_loki_logger", "LOKI_URL"]
