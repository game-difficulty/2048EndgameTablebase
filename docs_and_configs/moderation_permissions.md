# Main-Site Moderation

The main-site moderator role is `users.role = 'moderator'`. It is deliberately
not the legacy `admin` role used by competition administration and sponsor
presentation. Sponsorship remains a separate entitlement.

## Permissions

- Main-site owners retain the existing `ADMIN_ALLOWED_IDENTITIES` authority.
- Owners appoint/revoke moderators through `POST /api/admin/users/{id}/moderator`.
  This endpoint only transitions `user` and `moderator`; it cannot overwrite
  owner accounts or existing competition roles such as `organizer` and `admin`.
- Moderators can review all existing approval categories, review/reset profile
  changes, and disable/re-enable ordinary and supporter accounts.
- Moderators cannot act on themselves, owners, other moderators, or legacy
  `admin` accounts. Protected applications remain visible but read-only.
  Bulk profile review skips protected accounts. Owners handle those cases.
- Owner-only operations remain owner-only: balances/sponsorship adjustments,
  managed account password resets, global statistics, live controls/batch voids,
  and tablebase worker diagnostics.
- Tournament, forum and ranking permissions are unchanged. Becoming a moderator
  does not confer global organizer/referee authority or supporter presentation.

## API and UI

`backend/auth/management.py` defines the shared management permission checks.
Authenticated user payloads expose `management.owner` and `management.moderate`.
The shared account menu uses these capabilities on main, live and Play pages.

`GET /api/admin/permissions` rechecks the session and current account role.
The shared management page loads `/overview` only for owners. Moderators use
`GET /api/admin/users`, which omits balances, session/activity counts, global
statistics and total user counts. Pagination returns `has_more` instead.
Every write is separately authorized on the server; hiding UI is not security.
Existing sessions lose moderator API access on the next request after revocation.

## Allowance and Audit

Moderators have a 131,072-Token weekly free allowance. Existing semantics apply:
top up to the cap every seven days, not an additive weekly payment. This replaces,
rather than stacks with, the supporter/invited/public allowance.

Appointment and revocation reset free Tokens to the new role's weekly cap and
start a new seven-day interval. Permanent Tokens and sponsorship entitlements are
unchanged. Repeating an already-applied role update is a no-op, not another grant.
Role and account status changes are recorded in `management_audit`; allowance
changes also use the existing Token ledger. Approval/profile decisions retain
their existing operator audit records.

## Release Notes

Deploy the shared backend changes to both the main backend and the independent
Play backend, which also serves approval APIs. Run the normal auth database
initialization to create `management_audit` idempotently. No users are promoted
automatically. Frontend assets must be rebuilt for main/live/Play together.
Do not deploy unrelated competition or forum work to enable this feature.

Focused regression command:

```powershell
python -m unittest tests.test_moderator_permissions tests.test_admin_approval_transactions
```
