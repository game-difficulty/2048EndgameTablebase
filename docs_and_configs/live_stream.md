# AI live stream V1

## Architecture

One Python process on the local table machine owns the game. It uses the desktop
`Dispatcher`, `CoreAILogic`, and native `ai_core` library directly. No browser AI,
Node, cloud search, per-move table RPC, speculative prefetch, or viewing charges.
The publisher initiates an authenticated outgoing WebSocket to the cloud.

The independent Vite entry is `frontend/live/index.html`, available locally at
`/live/`. Production uses `live.2048tables.online`, not a main-site tab. The cloud
does only cheap exact-board reconstruction, broadcasting, bounded persistence,
and interaction. All viewers observe the same run and sequence.

## Wire protocol

WebSocket binary step: little-endian `uint32 sequence + uint32 delta_ms + uint8 move`.
The move byte follows RPL1: bits 0-1 direction (up/right/down/left), bits 2-5 cell,
bit 6 value (0=2, 1=4). Total: **9 application bytes per step**, including timing and
gap detection. At 12.5 moves/sec this is 112.5 bytes/sec/viewer, before WebSocket/TLS/TCP
overhead; not a measured billable-bandwidth guarantee. WebSocket compression is
unnecessary for such tiny frames. No score, board, name, or source repeats per step.

JSON messages are separate: snapshot/restart, decision-source changes, presence
(5 seconds), completed-game summary, chat, and aggregated likes. All transport
ordering is on one socket. Spectators never acknowledge individual moves.
An initial/reconnect snapshot contains exact tile values, score, elapsed time,
milestones, and sequence. A slow spectator has a bounded 32-message send buffer;
overflow replaces its backlog with the latest snapshot. Hidden tabs disconnect
and recover a snapshot when visible, rather than accumulating animations.

The runner sleeps only for the remaining part of its configured 80ms interval.
Slow AI searches may take longer. Game over waits 10-20 seconds. A disconnected
publisher triggers a separate offline overlay (up to 20 seconds for a silent
network failure), leaving history, stats, chat and links usable.

## Native runner

### Weekly statistics

The footer counts completed games by their finish time, Monday 00:00 through the
following Monday 00:00 in UTC+8. Counts and score totals aggregate `live_days`;
the exact score median uses compact `live_scores` rows, independent of the
200-replay/100MB retention limit. An even count averages the middle two scores.
The score ledger also keeps completion retries idempotent after replay pruning.

Startup backfills scores from retained replays without counting them again.
If older replay pruning has already removed scores, that week's median is null
(shown as a dash), not the median of an incomplete sample. Future complete weeks
have an exact median. Existing daily counters remain for backward compatibility.
The maintenance loop broadcasts a summary at a week boundary even while paused.

### Shared main-site login

Main-site and live-site sessions share the HttpOnly, Secure, SameSite=Lax cookie
`tb_shared_session` on the parent domain. `AUTH_SHARED_COOKIE_DOMAIN` defaults to
`2048tables.online`; only the parent, `www`, and `live` hosts issue it. Localhost
and other hosts keep the original host-only cookie. An empty setting disables
sharing. All subdomains of this parent must remain trusted; do not host untrusted
applications on that cookie domain.

New logins and registrations work on a first visit to the other site. Existing
main-site users refresh the main site once to migrate their cookie automatically;
the session and expiry are unchanged. HTTP and WebSocket authentication prefer
the shared session over old per-host credentials. Logout revokes that session and
clears the shared cookie. The live page refreshes identity when its tab becomes
visible again. There are no cross-origin redirects, iframe bridges or URL tokens.

### Administrator pause / resume

The main-site administrator page controls the livestream through authenticated
`GET /api/admin/live` and `POST /api/admin/live {"enabled": false|true}`.
The desired state and revision persist in `live_control` in the live database.
Repeated requests are idempotent; reconnecting or restarting the server does not
clear a pause. Starting while the local computer is offline only saves the intent.
It cannot launch the local process remotely.

The publisher advertises `control_version: 1`, receives the current control in
the handshake and subsequent `control` messages, and replies with `control_ack`.
Pause keeps the socket heartbeat and current run, seed and checkpoint. An ongoing
native query may finish, but its result is discarded if the control revision has
changed. Steps already sent before the ACK are accepted in socket order. No new
steps or games are sent after the pause ACK, and paused time is not game time.
The separate tablebase Worker serving interactive users is unaffected.

Deploy the server and frontend, then update/restart the local livestream runner
once to load this support. Subsequent pause/resume actions do not restart it.
The administrator page warns about older runners instead of pretending a pause
was applied. Viewers see a distinct paused overlay; chat and history remain usable.

Run from the cloud checkout with its normal Python environment:

```powershell
$env:LIVE_PUBLISH_TOKEN = '<secret from your secret manager>'
C:/Anaconda/python.exe tools/live_runner.py --engine-root C:/Apps/2048endgameTablebase/src --threads 1 --interval 0.08 --log-file data/live-runner.log
```

Never put the token in the URL, source control, or logs. Both publisher and server
must use the same cryptographically random token of at least 32 characters.
`--tables path.json` optionally supplies an explicit map of full pattern to
`[path, dtype]` pairs; otherwise the local desktop configuration is used. Example:

```json
{"free10_512": [["C:/2048_tables/free10-512", "uint32"]]}
```

The configuration is
changed only in memory, never saved back to desktop settings. Paths stay local.

The native search is limited to one thread by default. Local table reads may still
compete with the existing Worker for disk I/O: keep process priority low and
increase `--interval` if user queries deteriorate. This V1 does not claim to have
preemptive priority over reads in the separate Worker process.

Use a Windows scheduled task (at startup, restart on failure, hidden window) or an
existing service wrapper. No such task is installed automatically by this change.
The runner checkpoint defaults to `data/live-runner.json`, updated every 5 seconds
and at new-game/end. No per-step console log. Connection failure logging backs off
to once per minute. Native exceptions pause/reconnect; they are never fake deaths.

The record uses exact game rules, unlike `AItest.py`'s historical 65K substitution.
Native search still receives its supported packed projection, while decisions are
applied to the exact board. Illegal native decisions pause instead of corrupting
the game. Replay uses the existing 2048next compact format, supporting 65K+ tiles.

## Cloud deployment

Set in the cloud service environment:

```text
LIVE_PUBLISH_TOKEN=<random secret>
LIVE_DB_PATH=/opt/2048tables/data/live.sqlite3
LIVE_MAX_VIEWERS=200
LIVE_MUSIC_URL=
```

One backend process owns the channel. Multiple independent uvicorn workers are not
supported in V1. API: GET `/api/live/state`, WS `/api/live/watch`, authenticated
publisher WS `/api/live/publish`, POST `/api/live/chat`, POST `/api/live/like`,
GET `/api/live/replays/{id}`. The producer credential cannot be used by browsers.
Publisher exclusivity is enforced by the single cloud process.

Build with `npm run build`; deploy the complete generated frontend assets and both
HTML entries. The added shared animation-duration prop defaults to the old 300ms;
only live boards request a 45ms animation. Gamer's move implementation has been
extracted unchanged to `classicMove.js`, shared by the live frontend.

Production uses `deploy/live.nginx.conf` with a dedicated Let's Encrypt certificate.
Do not activate a config pointing at certificate paths that do not exist. Both
viewer and authenticated publisher sockets use the live subdomain. Keep proxy request/connection
limits independent from game/Worker limits. Never trust public forwarded-IP
headers: the sample proxy overwrites them. If behind a trusted CDN, configure real
IP handling explicitly. The deployment template is not applied automatically.

The deployed virtual host trusts only Cloudflare's published proxy networks before
using the visitor address for rate limits. Publisher connections have their own
limit zones. `tools/start_live_runner.ps1` starts the local runner hidden, at
BelowNormal priority, with one search thread and an 80ms minimum step interval.
Its credential is loaded from the user's protected `.config/2048tables/live.env`,
outside the repository. The main runner log rotates at 512KB with one backup.
This launcher does not install a Windows startup task.

Users may watch anonymously and chat after automatic guest-session creation.
Main-site and live logins share an HttpOnly, Secure, SameSite=Lax cookie on
`2048tables.online`. Existing host-only sessions migrate when `/api/auth/me` is
requested, without extending their lifetime. Local development retains host-only
cookies. Account data, avatars and supporter badges use the existing database.

The local live-only table allowlist is `live_ai_tables.local.json`. It excludes
`free10_128`, `444_1024` and `444_2048`; this does not change the public tablebase
Worker catalog. Gamer excludes these three tables by default too; explicit user
selections, including Select All, can enable them again.
The runner uses a 50ms minimum for search and 80ms for table decisions. It only
acknowledges broadcast control when the server advertises that capability, so
restarting it before a server upgrade remains safe.

## Retention, statistics and interaction

Current checkpoint is persisted every 5 seconds. Reconnect restores from whichever
side has the longer matching prefix; divergence is rejected. Stats/finished-game
records are idempotent by run ID. History keeps at most 200 games and 100MiB of replay
text; the small per-day aggregates and all-time best remain. SQLite reuses freed
pages; no repeated full VACUUM. Only natural deaths count as completed games.
Daily boundary is UTC+8. 32K/64K counts mean completed games whose maximum tile
reached that threshold. Average excludes the live, unfinished game.

Recent history links to `https://2048tables.online/verse-replay/?live=<id>`.
Expired links report that the replay is unavailable. A local preview can replace
the link origin when testing the same API locally.

Chat: last 100 messages in memory; 32 Unicode code points, 5 messages/minute/actor,
plus a 20/minute/IP safety cap. No HTML rendering or control characters. Recent
chat is intentionally ephemeral across cloud process restart. Guests reuse the
existing signed identity; a new browser identity cannot bypass the IP cap.
Likes: 20/minute/actor, 60/minute/IP, persisted/broadcast in 5-second batches.
Every click gets immediate optimistic feedback. Clicks are collected for 500ms
and POSTed as a count (1-20); clicks during an in-flight request remain queued for
the next batch. Only one request is in flight. No automatic retry can duplicate
an accepted batch. Failed batches roll back without dropping later clicks.
The POST response includes the accepted total, without waiting for a broadcast.
Animations have bounded independent particles and do not restart on every click.
Music is off by default. Two CC0 tracks ship under `frontend/public/live-music/`
(about 4MB combined, 96kbps MP3); source links and credits accompany them. Audio
loads only on a play request. List playback wraps to the first track, with previous,
next, selection and volume controls. Multiple local files may be added (never
uploaded); duplicates are skipped and object URLs are released on removal/unmount.
The optional `LIVE_MUSIC_URL` adds a channel track. Music is not synchronized.

## Token gifts

Logged-in viewers can send gifts without changing the AI game, sponsorship status,
or donation totals. No money or Tokens are transferred to a broadcaster account.
The existing bonus-first token ledger applies, including the global price multiplier
but no table multiplier. The catalogue is read from `cloud_token_costs.json`:

| Gift | Base Tokens |
| --- | ---: |
| A Little Two | 2 |
| Little Heart | 8 |
| Flowers | 8 |
| Nice Four | 4 |
| Ace Dealer / KLBM | 32 |
| 666 | 66 |
| BUG | 8 |
| Mind Expanded / The Button / Whale | 16 |
| What (Moai) | 10 |
| What Does It Mean? / RIP / Easy as Tea | 16 |
| Chicken Workout / Serious Splits | 16 |
| Nice Merge | 64 |
| Coffee | 128 |
| Fireworks | 512 |
| Brilliant Move | 1024 |
| final 1k | 1024 |
| 2048! | 2048 |
| 32K? Too Easy | 32768 |
| 65K Legend | 65536 |

`backend/live/gifts.py` creates three small tables in the **authentication database**:
`live_gift_orders`, `live_gift_preferences`, and `live_gift_daily`. Orders, the
existing token ledger, idempotency rows, daily budget and broadcast outbox are
committed in one SQLite transaction. A quote contains exact quantity totals and
a price version. Stale quotes are rejected without charging. Request IDs are UUIDs;
retries must match the original user and payload. Offline streams reject new gifts
but still return receipts for committed retries.

API under `/api/live/gifts`: GET `catalog`, GET `me`, POST `preferences`, POST `send`,
GET `orders/{request_id}`. Sends allow 1-1000 units, capped at 5 requests/sec and
30/minute/user, plus 90/minute/IP. Preferences include an optional daily Token budget
and supporter entrance opt-out. Days follow UTC+8.

The send button shows the full charge. Orders of 2048 Tokens or more require an
additional confirmation. Quick send needs an explicit session budget, reserves
queued amounts, and keeps at most five pending requests. Uncertain network results
retain the same purchase ID in sessionStorage and expose a check/retry action.
The client never invents a successful global event before server confirmation.

Outbox delivery runs immediately after purchase and retries every second. Event IDs
deduplicate delivery. Gifts within five seconds by the same user for the same item
share one combo. Stale outbox events older than 30 seconds remain in history without
replaying effects. Animation payloads older than 30 days are cleared; compact order
receipts and billing records remain. No per-step or full-payload gift logs.

Gift events also occupy one normal chat row per combo, updating its count in place.
The server restores gift chat rows from existing receipts on startup; reconnects
merge HTTP history and WS events by combo ID without replaying effects.

Gift artwork references are retained in `design/live_gifts/references` beside this
document. Runtime WebP assets live in `frontend/public/live-gifts`; catalog and chat
always use static posters. Only the active supporter banner loads animated WebP
(button, tea, moai). Whale spouts and the two chicken motions use bounded CSS/SVG
animations. Both supporter tiers qualify; ordinary users get the static banner.
The four ceremony gifts (final, 2048, crown, legend) have separate stamp/merge/crown
scenes. At most one ceremony plays alongside the banners, with no pointer capture.
Simple/off modes and reduced-motion preferences disable these scenes and motion
assets. Combo quantity changes never remount or restart the active animation.

Top effects overlay the stage, with two desktop lanes or one mobile lane,
bounded 20-item priority queue and 2.5/3.5/5-second durations. A combo does not restart
the entrance animation and occupies a lane for at most eight seconds. Full/simple/off
and reduced-motion are supported. Supporter/admin entrances share the normal badge,
are deduplicated across live sockets, and have a persistent 30-minute account cooldown.
Gift history merges combo rows; each user can inspect their last 50 purchases.

### Supporter presentation levels

The live-only `supporter_level` is derived from confirmed cumulative CNY amounts
in `admin_topup` ledger metadata (`payment_amount_cny`, positive paid delta).
An indexed subset of top-up rows is read only for entrance/chat/gift interactions,
never on AI steps. No new payment amount is exposed to viewers and no balance or
gift consumption is used to infer sponsorship. Balance-setting events are excluded.

- Level 1: at least CNY 9.90. Teal name/gem; entrance goes to chat only.
- Level 2: at least CNY 99.00. Gold name/crown; entrance also occupies a top banner
  for five seconds. Chat gets a restrained gold background/side rule; gift banners
  get gold corner hatching and an extra 1.5 seconds (the eight-second combo cap remains).
- Existing supporter/admin identities without payment metadata retain level 1;
  admin status alone never grants level 2. No change to global entitlement tiers,
  weekly allowances, gift prices, chat limits, entrance cooldown or opt-out.

Both entrance levels remain in the bounded 100-entry chat history. Reconnect history
does not replay banners; reduced-motion/simple/off still applies to top effects.

## Viewer-funded red envelopes

The expanded gift drawer contains a separate red-envelope action, not a bulk gift.
Logged-in viewers can transfer 1,000-50,000 **paid** Tokens into 5-20 shares. The
full amount is debited once on final confirmation; the per-sender cooldown is 15
seconds. Weekly allowances, pricing multipliers and ordinary gift budgets do not
apply. Transfers do not grant gift contribution or sponsorship status.

`backend/live/red_envelopes.py` owns the persistent global FIFO queue in the auth
database. Only the head can be claimed. Its clock starts on activation, stays on
screen for at least 15 seconds even if exhausted, and expires at 60 seconds.
Unclaimed shares return to the sender automatically, including after disconnects
or a server restart. A queued envelope gets its own full minute when activated.
The sender cannot claim their own envelope; each other logged-in viewer with a
fresh room heartbeat can claim once. No AI publisher connection is required.

Random mode samples a uniform positive integer composition and shuffles it using
the system CSPRNG, giving every claim position the same expectation. Allocations
never leave the server. Equal mode floors each share; its remainder is shown
before confirmation and is not refunded. Only unclaimed distributable shares are
refunded on expiry.

SQLite IMMEDIATE transactions couple sends, claims and refunds to their paid
Token ledger entries (`live_red_send`, `live_red_award`, `live_red_refund`). Sender
request IDs and claimant uniqueness prevent double debit/credit. A persisted
outbox restores chat announcements; snapshots restore the active envelope. The
refund task runs independently of ordinary gift delivery. The browser retains
uncertain send requests across refresh and retries with the same immutable ID.

The original `红包.gif` is retained in `design/live_gifts/references`; transparent
static/animated WebP assets are in `frontend/public/live-gifts`. The envelope
appears below lucky bags. Clicking its chat announcement reopens its status and
the viewer's own award. Guests are prompted to sign in. Reduced motion uses the
static poster.

## Verification

```text
python -m unittest tests.test_live tests.test_live_routes tests.test_live_gifts
python -m unittest tests.test_live_red_envelopes
node --test tests/liveEngine.test.js tests/liveInteractions.test.js tests/liveGifts.test.js tests/boardFrame.test.js tests/verseReplayCore.test.js
npm run build
```

Before production: verify guest issuance behind the new host, HTTPS login/logout,
two simultaneous viewers, publisher disconnect/reconnect, process restart, natural
death/restart, replay from history, short/long native searches, and local disk impact.
Use a separate development database and publisher token for smoke tests.
