from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "competition.backend.app:app",
        host=os.getenv("COMPETITION_HOST", "0.0.0.0"),
        port=int(os.getenv("COMPETITION_PORT", "8001")),
        log_level=os.getenv("COMPETITION_LOG_LEVEL", "info"),
        reload=os.getenv("COMPETITION_RELOAD", "0") == "1",
    )


if __name__ == "__main__":
    main()

