# Daily Account Activity

Daily active accounts use Beijing calendar days (UTC+8). Visitors without an
authenticated account are excluded. The admin chart counts distinct user IDs
across all sites, so an account visiting several sites counts only once per day.

## Shared Integration

`backend/auth/activity_middleware.py` implements the statistics policy for HTTP
and WebSocket traffic. Authentication adapters call `bind_activity_account`
only after validating a real account, exposing its ID in connection state.
Feature endpoints must not insert daily activity rows themselves.

The middleware is installed in:

- `backend/app.py`: main and live sites; source is selected from the host.
- `backend/human_play/production_app.py`: Play site, source `play`.
- `competition/backend/app.py`: competition site, source `tournament`.

HTTP activity is recorded when an authenticated request produces a response
below status 500 (including an authenticated user's rejected operation, such as
an invalid board or insufficient balance). Failed authentication does not mark
an account. Login/register/password-reset handlers also mark their newly
authenticated account. WebSocket activity is recorded on incoming and outgoing
application messages, including cross-day traffic on an existing connection.
Opening an anonymous socket or closing a socket alone does not count. Competition
development identities never count as real authenticated activity.

## Cost and Failure Handling

`record_daily_visit` caches committed `(database, day, user, site)` entries in a
bounded process-local cache (16,384 entries). Concurrent requests/processes may
attempt the initial insert; the SQL primary key makes it idempotent. No lock is
held across database I/O. Cache misses write in a thread instead of blocking the
ASGI event loop. A WebSocket additionally remembers its last recorded day/account.

Database failures do not fail gameplay or authentication. They are logged and
retried on later requests; a failed connection attempt has a 60-second retry
backoff. Failed writes are never cached as successful. Restarting a process only
causes another idempotent insert, not duplicated accounts.

## Adding a Site or Authentication Method

For a new application, install `DailyActivityMiddleware`, add its site to
`SITES`, and select it explicitly or add its hostname to `site_from_host`.
For a new authentication adapter, mark the validated account on the request or
WebSocket. Do not add statistics code to individual game/chat/query handlers.
No schema migration is required for `tournament`: the existing site column is text.

Client-only offline activity cannot be inferred by the server until an
authenticated request/message occurs. Mere background maintenance, token grants,
static assets and unauthenticated public requests are not account activity.

Tests: `python -m unittest tests.test_daily_activity -v`.

## Historical Backfill and Existing Timer

`daily_user_activity(day, user_id, site)` remains the durable source. The
idempotent backfill script also reads login sessions, quota operations, ranked
starts, minigame starts, battle membership/chat, Live gift and draw participation
from the auth database. It reads Play run starts and received move batches from
`HUMAN_PLAY_DB`. Automated grants, awards and settlements are excluded.

The existing `deploy/2048tables-daily-activity.service` and its timer fill missed
activity for the chart's 14-day window at 00:10 Beijing time. The units load the
Cloud and optional Play environment files and pass the Play database explicitly.
Keep this fallback; it does not replace request-time instrumentation. This change
cannot backfill historical competition visits that were never recorded.
Passing `--competition-db` additionally reads competition commands, enrollment
and roster audit actors, and retained practice-best submissions. It does not
infer activity from scheduled rosters, automatic results, or event creation.
Practice-best records preserve only retained personal bests, not every past run.

Manual backfill:

```sh
cd /opt/2048tables/app
/opt/2048tables/venv/bin/python -m scripts.refresh_daily_activity --days 14 --play-db /var/lib/2048tables/play/human.sqlite3 --competition-db /var/lib/2048tables/competition-test/competition.sqlite3
```

Older page views and Live viewers without recorded actions cannot be reconstructed.
Pre-instrumentation values are lower bounds; compare historical trends with care.
