"""Live-only presentation levels derived from confirmed payments, never balances."""
import json
from contextlib import nullcontext
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from backend.auth.db import auth_db


def supporter_level(user, db=None):
    if not user:
        return 0
    legacy_supporter = user.get('role') == 'admin' or (user.get('entitlements') or {}).get('tier') == 'supporter'
    cents = 0
    if user.get('id'):
        with (nullcontext(db) if db is not None else auth_db()) as connection:
            rows = connection.execute('''SELECT metadata_json FROM token_ledger
                WHERE user_id=? AND event_type='admin_topup' AND paid_delta_units>0''', (user['id'],))
            for row in rows:
                try:
                    raw = json.loads(row['metadata_json'] or '{}').get('payment_amount_cny')
                    if raw is None or isinstance(raw, bool):
                        continue
                    amount = Decimal(str(raw))
                    if not amount.is_finite() or amount <= 0:
                        continue
                    if amount >= 99:
                        return 2
                    cents += int((amount * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
                    if cents >= 9900:
                        return 2
                except (ValueError, TypeError, AttributeError, InvalidOperation):
                    continue
    return 1 if cents >= 990 or legacy_supporter else 0


def public_actor(user, db=None):
    user = user or {}
    level = supporter_level(user, db)
    return dict(name=user.get('display_name') or 'User',
                avatar_url=(user.get('profile') or {}).get('avatar_url'),
                supporter=level > 0, supporter_level=level)
