from __future__ import annotations

from typing import Any

import numpy as np
from engine_core.VBoardMover import encode_board

from ..session import safe_hex, u64
from .errors import RemoteTablebaseProtocolError
from .registry import remote_worker_registry


class RemoteAnalysisBookReader:
    def __init__(
        self,
        *,
        full_pattern: str,
        pattern: str,
        target: str,
        use_variant: bool,
    ) -> None:
        self.full_pattern = full_pattern
        self.pattern = pattern
        self.target = str(target)
        self.use_variant = bool(use_variant)
        self._cache: dict[int, tuple[dict[str, Any], str]] = {}

    @staticmethod
    def _encoded(board: Any) -> int:
        if isinstance(board, (int, np.integer)):
            return u64(board)
        return u64(encode_board(np.asarray(board)))

    def preload(self, boards: list[int], *, batch_size: int = 256) -> None:
        unique = [u64(board) for board in dict.fromkeys(boards) if u64(board) not in self._cache]
        for offset in range(0, len(unique), max(1, int(batch_size))):
            batch = unique[offset : offset + batch_size]
            response = remote_worker_registry.run_from_worker_thread(
                remote_worker_registry.lookup_batch(
                    full_pattern=self.full_pattern,
                    pattern=self.pattern,
                    target=self.target,
                    boards=[safe_hex(board) for board in batch],
                    use_variant=self.use_variant,
                    board_is_lookup=True,
                ),
                timeout=max(65.0, min(605.0, len(batch) * 2.0 + 5.0)),
            )
            items = response.get("items")
            if not isinstance(items, list):
                items = response.get("results")
            if not isinstance(items, list) or len(items) != len(batch):
                raise RemoteTablebaseProtocolError(
                    "Remote tablebase returned an invalid analysis batch."
                )
            for board, item in zip(batch, items):
                if not isinstance(item, dict):
                    raise RemoteTablebaseProtocolError(
                        "Remote tablebase returned an invalid analysis result."
                    )
                raw_results = item.get("results", item)
                if not isinstance(raw_results, dict):
                    raw_results = {}
                self._cache[board] = (
                    dict(raw_results),
                    str(item.get("dtype") or response.get("dtype") or "uint32"),
                )

    def move_on_dic(self, board, pattern: str, target: str, full_pattern: str):
        encoded = self._encoded(board)
        cached = self._cache.get(encoded)
        if cached is None:
            response = remote_worker_registry.run_from_worker_thread(
                remote_worker_registry.lookup(
                    full_pattern=self.full_pattern,
                    pattern=self.pattern,
                    target=self.target,
                    board=safe_hex(encoded),
                    use_variant=self.use_variant,
                    board_is_lookup=True,
                )
            )
            results = response.get("results")
            cached = (
                dict(results) if isinstance(results, dict) else {},
                str(response.get("dtype") or "uint32"),
            )
            self._cache[encoded] = cached
        return dict(cached[0]), cached[1]
