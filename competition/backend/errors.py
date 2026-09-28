from __future__ import annotations


class CompetitionError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

    @property
    def detail(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}

