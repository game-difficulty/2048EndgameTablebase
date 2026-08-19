"""Local tablebase worker for serving allowlisted tables over WSS."""

from .config import WorkerConfig, load_worker_config

__all__ = ["WorkerConfig", "load_worker_config"]
