"""
Structured logging configuration for the RAG system.
Provides console and rotating file handlers with consistent formatting.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from config import settings


def setup_logger(name: str = "mike") -> logging.Logger:
    """
    Set up and return a configured logger.

    Args:
        name: Logger name for the application module.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if settings.log_to_file:
        os.makedirs(settings.log_folder, exist_ok=True)
        log_file = os.path.join(settings.log_folder, "app.log")

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_logger()
