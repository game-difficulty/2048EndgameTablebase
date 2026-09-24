"""Short, stateless online permits; the current owner/epoch/eligibility is still read."""
import hashlib
import base64
import hmac
import time
from functools import lru_cache

from .store import database, db_path


@lru_cache(maxsize=8)
def _key(path):
    with database() as db:
        return bytes(db.execute("SELECT secret FROM human_keys WHERE id=1").fetchone()[0])


def _signature(run, expiry):
    # Fixed database identities plus length-delimited encoding prevent ambiguity.
    import json
    fields = [run[k] for k in ('id', 'user_id', 'browser', 'writer', 'epoch')]
    message = json.dumps([*fields, expiry], separators=(',', ':')).encode()
    return hmac.new(_key(str(db_path().resolve())), message, hashlib.sha256).digest()


def issue(run, until):
    expiry = str(int(until * 1000))
    return expiry + '.' + base64.urlsafe_b64encode(_signature(run, expiry)).decode().rstrip('=')


def valid(run, token, now=None):
    try:
        expiry, signature = token.split('.')
        seconds = int(expiry) / 1000
        current = time.time() if now is None else now
        # Existing open tabs may carry the earlier hexadecimal signature for 12 seconds.
        supplied = bytes.fromhex(signature) if len(signature) == 64 else base64.b64decode(signature + '=', altchars=b'-_', validate=True)
        return current <= seconds <= current + 13 and hmac.compare_digest(supplied, _signature(run, expiry))
    except (ValueError, TypeError, AttributeError):
        return False
