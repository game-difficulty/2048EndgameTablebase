import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("forum.backend.app:create_app", factory=True,
                host=os.getenv("FORUM_HOST", "127.0.0.1"),
                port=int(os.getenv("FORUM_PORT", "8002")))
