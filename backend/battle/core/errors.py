from __future__ import annotations

from typing import Any


class BattleServiceError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        *,
        extra: dict[str, Any] | None = None,
    ):
        self.code = str(code)
        self.status_code = int(status_code)
        self.extra = dict(extra or {})
        super().__init__(message)

    @property
    def detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), **self.extra}
