import sys
import threading
import traceback
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


def publish_frontend_exception(
    title: str,
    exc: BaseException | None = None,
    exc_info: tuple[type[BaseException], BaseException, object] | None = None,
) -> None:
    if exc_info is None:
        if exc is not None:
            exc_info = (type(exc), exc, exc.__traceback__)
        else:
            current = sys.exc_info()
            if current[0] is None or current[1] is None:
                publish_frontend_error(title, "")
                return
            exc_info = (current[0], current[1], current[2])

    exc_type, exc_value, exc_traceback = exc_info
    formatted = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    ).strip()
    publish_frontend_error(title, formatted)


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
