import threading
from collections import deque
from typing import Callable

ErrorPayload = dict[str, str]
ErrorDispatcher = Callable[[ErrorPayload], None]

_dispatcher: ErrorDispatcher | None = None
_pending_errors: deque[ErrorPayload] = deque(maxlen=16)
_lock = threading.Lock()


def publish_frontend_error(title: str, message: str) -> None:
    payload: ErrorPayload = {"title": title, "message": message}

    with _lock:
        dispatcher = _dispatcher
        if dispatcher is None:
            _pending_errors.append(payload)
            return

    try:
        dispatcher(payload)
    except Exception:
        with _lock:
            _pending_errors.append(payload)


def register_frontend_error_dispatcher(dispatcher: ErrorDispatcher) -> None:
    pending: list[ErrorPayload]
    with _lock:
        global _dispatcher
        _dispatcher = dispatcher
        pending = list(_pending_errors)
        _pending_errors.clear()

    for payload in pending:
        try:
            dispatcher(payload)
        except Exception:
            with _lock:
                _pending_errors.appendleft(payload)
            break

