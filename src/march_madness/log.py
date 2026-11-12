"""Logging helpers."""

import logging

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(*, level: int = logging.INFO, console_width: int = 100) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                console=Console(width=console_width),
            )
        ],
    )
