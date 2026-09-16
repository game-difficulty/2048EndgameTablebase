# Live Room Layout and Contributions

The core board/timing/history layout is unchanged. A common-gift strip sits below
it; its chevron opens a temporary drawer over the stage's right side. Gift hover
or keyboard focus reveals Send 1, followed by 10/100/1000 and a custom 1-1000 input.
Touch activates the same controls by tapping the artwork, never by purchasing on
hover. All purchases show their price; totals of 2048 Tokens or more require a
confirmation unless an explicit quick-send budget has been authorized.

The right column contains the audience toggle, the website introduction and main
site URL, scrollable chat, and the message composer. Music is a compact horizontal
strip at the bottom. Narrow viewports retain the core page's existing responsive
behavior and put chat underneath rather than shrinking text.

## Audience

`GET /api/live/audience` returns only currently connected viewers, sorted with
level-2 supporters first, then descending contribution. Ties use a stable public
hash. Emails, IPs and internal actor/session credentials are never returned.
The component refreshes every five seconds only while expanded and visible.

Contribution = actual gift Tokens spent + floor(online watching seconds / 600)
+ min(10, accepted likes) + min(10, accepted chat messages).
Token amounts retain fractional units; gift credits are written in the same
transaction as the order and cannot be duplicated by retry or combo updates.
Watch time is counted once per authenticated user or guest identity across tabs,
in short aggregate ticks, only while the producer is online. Brief sub-tick visits
do not earn watch credit. No heartbeat log or per-step contribution record is kept.

A broadcast session spans AI game restarts and short producer outages. A producer
returning after more than 30 minutes offline starts a fresh session. The session
and aggregate rows persist in the auth database; previous aggregates are removed
when a new session begins. Closing a viewer connection removes it from the visible
roster but preserves its session contribution for rejoining.

The catalog retains exact totals for 1-10 for existing clients, and additionally
publishes integer base/global-multiplier units. Custom totals use the same half-even
rounding as the server, with BigInt on the browser. The server still verifies the
quote version, exact total, quantity, identity, budget and balance atomically.
