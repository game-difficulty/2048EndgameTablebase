from __future__ import annotations


class RemoteTablebaseError(RuntimeError):
    code = "REMOTE_TABLEBASE_ERROR"

    def __init__(self, message: str = "Remote tablebase request failed.") -> None:
        super().__init__(message)

    @property
    def payload(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


class RemoteTablebaseOffline(RemoteTablebaseError):
    code = "REMOTE_TABLEBASE_OFFLINE"

    def __init__(self) -> None:
        super().__init__("The selected tablebase is temporarily unavailable.")


class RemoteTablebaseTimeout(RemoteTablebaseError):
    code = "REMOTE_TABLEBASE_TIMEOUT"

    def __init__(self) -> None:
        super().__init__("The remote tablebase request timed out. Please retry.")


class RemoteTablebaseProtocolError(RemoteTablebaseError):
    code = "REMOTE_TABLEBASE_PROTOCOL_ERROR"

