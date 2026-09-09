"""Compatibility entry point for the packaged AI batch runner."""

import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.ai_batch import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
