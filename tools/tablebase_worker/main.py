from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .client import WorkerClient
from .config import DEFAULT_CONFIG_PATH, WorkerConfigError, load_worker_config
from .logging_setup import configure_logging
from .reader_pool import ReaderPool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2048 local tablebase WSS worker")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the private local worker JSON config",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate config and table paths without connecting or querying",
    )
    return parser


async def run_worker(config) -> None:
    reader_pool = ReaderPool(config)
    client = WorkerClient(config, reader_pool)
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(signal_name, client.stop)
    try:
        await client.run()
    finally:
        await reader_pool.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_worker_config(args.config)
    except WorkerConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    logger = configure_logging(config.log)
    if args.check_config:
        reader_pool = ReaderPool(config)
        for table_id, status in reader_pool.status().items():
            print(f"{table_id}: {'ready' if status['ready'] else status['error_code']}")
        asyncio.run(reader_pool.close())
        return 0 if len(reader_pool.ready_tables) == len(reader_pool.allowed_tables) else 1
    logger.info("Starting worker %s", config.worker_id)
    try:
        asyncio.run(run_worker(config))
    except KeyboardInterrupt:
        logging.getLogger("tablebase_worker").info("Worker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
