"""
Logging setup for MALS.

Provides a rich, colorful console output for the Conductor loop,
making it easy to follow the multi-agent collaboration in real-time.
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the MALS system.

    Uses Rich for colorful, structured console output.

    Args:
        level: The logging level (DEBUG, INFO, WARNING, ERROR).
    """
    console = Console(stderr=True)

    handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        handlers=[handler],
        force=True,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
