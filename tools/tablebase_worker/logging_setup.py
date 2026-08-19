from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .config import LogConfig


def configure_logging(config: LogConfig) -> logging.Logger:
    config.path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tablebase_worker")
    logger.setLevel(getattr(logging, config.level))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        config.path,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
