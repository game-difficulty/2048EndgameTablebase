"""Main-site moderation permissions, independent of tournament and forum roles."""
import os

from fastapi import HTTPException

MODERATOR_ROLE = 'moderator'
DEFAULT_ALLOWED_IDENTITIES = ('user0', 'assweeass@163.com')


def allowed_identities():
    values = [part.strip().lower() for part in os.getenv('ADMIN_ALLOWED_IDENTITIES', '').split(',') if part.strip()]
    return set(values or DEFAULT_ALLOWED_IDENTITIES)


def is_owner(user):
    if user is None:
        return False
    user = dict(user)
    allowed = allowed_identities()
    return (str(user.get('email') or '').strip().lower() in allowed
            or str(user.get('display_name') or '').strip().lower() in allowed)


def management_permissions(user):
    user = dict(user or {})
    owner = is_owner(user)
    return {'owner': owner, 'moderate': owner or user.get('role') == MODERATOR_ROLE}


def can_moderate_target(actor, target):
    if target is None:
        return False
    if is_owner(actor):
        return True
    target = dict(target)
    return (management_permissions(actor)['moderate']
            and int(actor['id']) != int(target['id'])
            and not is_owner(target)
            and target.get('role') not in {'admin', MODERATOR_ROLE})


def require_moderation_target(actor, target):
    if target is None:
        raise HTTPException(404, 'User not found.')
    if not can_moderate_target(actor, target):
        raise HTTPException(403, 'This account must be reviewed by the site owner.')
