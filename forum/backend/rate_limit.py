import hashlib
import hmac
import ipaddress
from datetime import datetime, timezone
from .db import one


def client_ip(request, settings):
    peer = request.client.host if request.client else "unknown"
    try:
        trusted = any(
            ipaddress.ip_address(peer) in ipaddress.ip_network(net)
            for net in settings.trusted_proxies
        )
        if trusted:
            # Trust only a single proxy-written address; never blindly take the leftmost XFF.
            real = request.headers.get("x-real-ip", "")
            return str(ipaddress.ip_address(real)) if real else peer
    except ValueError:
        pass
    return peer


def allow(engine, ip, write, secret):
    day = datetime.now(timezone.utc).date().isoformat()
    key = hmac.new(
        secret.encode(), f"{day}|{ip}|{write}".encode(), hashlib.sha256
    ).hexdigest()
    with engine.begin() as conn:
        row = one(
            conn,
            """INSERT INTO forum_ip_windows(key,count) VALUES(:k,1) ON CONFLICT(key) DO UPDATE SET
            count=CASE WHEN forum_ip_windows.window_started<now()-interval '1 minute' THEN 1 ELSE forum_ip_windows.count+1 END,
            window_started=CASE WHEN forum_ip_windows.window_started<now()-interval '1 minute' THEN now() ELSE forum_ip_windows.window_started END
            RETURNING count""",
            k=key,
        )
    return row["count"] <= (120 if write else 600)
