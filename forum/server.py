import os
import logging

import uvicorn

if __name__ == "__main__":
    request_logger = logging.getLogger("forum.backend.app")
    request_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    request_logger.addHandler(handler)
    request_logger.propagate = False
    uvicorn.run(
        "forum.backend.app:create_app",
        factory=True,
        proxy_headers=False,
        host=os.getenv("FORUM_HOST", "127.0.0.1"),
        port=int(os.getenv("FORUM_PORT", "8002")),
    )
