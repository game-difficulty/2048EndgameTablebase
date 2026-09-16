# Live Lucky Bags

## Rules

| First merged tile in a run | Pool | Winners | Per winner |
| --- | ---: | ---: | ---: |
| 32768 | 10,000 Tokens | Up to 10 | 800-2,000 Tokens |
| 65536 | 100,000 Tokens | Up to 10 | 8,000-20,000 Tokens |

Both bags open for 180 seconds and remain visible for 180 seconds after drawing.
Logged-in users enter once per bag. Multiple browser connections do not add entries.
Winners are sampled uniformly without replacement from eligible participants.
Distribute `min(pool, winner_count * maximum)` whole Tokens, randomly split within
the per-person bounds. With ten winners the entire pool is distributed. With five
or fewer, each winner receives the maximum and any unused pool is not awarded.

Eligibility requires a live authenticated viewer connection at draw time: connected
before the entry deadline, still registered, and with a heartbeat within 35 seconds.
Closed/hidden pages disconnect through the existing viewer lifecycle. A connection
opened after the deadline cannot make an absent participant eligible. If the service
is offline at the deadline, late reconnects do not qualify for that expired draw.
Producer outages do not cancel already-open bags or pause their countdowns.

## Server

`backend/live/lucky_bags.py` owns persistent rules, entries, drawing and accounting.
The existing authenticated producer's validated `LiveRun.nodes` supplies milestones.
The hub checks these in memory every second; ordinary AI moves add no database work
and no extra per-step payload. Unique `(run_id, milestone)` keys prevent duplicate
bags after reconnects or checkpoint recovery. Multiple overlapping bags are retained.

`BEGIN IMMEDIATE` covers winner selection, paid-balance credits, ledger entries and
the completed draw marker. A failure rolls back all awards, and repeated draws are
no-ops. Clients never claim an award and never send their prize amounts.
Ledger event `live_lucky_award` credits paid Tokens without counting as sponsorship,
gift contribution or Token consumption. Legacy entitlement backfill excludes these
reward-bearing accounts; actual admin sponsorship continues to set entitlements.

The hub's cached public bag state is included in snapshots and changed-state WS
events. Counts update on entries; outcomes update on drawing. APIs:

- `GET /api/live/lucky-bags`: no-store public list, plus the authenticated user's
  entry/result only. It does not expose other participants' identities.
- `POST /api/live/lucky-bags/{id}/join`: authenticated, same-origin, requires room
  presence, rate limited to 12/minute per account. Retries are idempotent.

Participation rows expire after 30 days; the small trigger keys and accounting ledger
remain. No per-step or participant-body logs are added.

## Browser

The floating entry is in the core's upper-left area. Dismissal is scoped to a bag ID
in sessionStorage; the gift strip keeps its entry at the far left. Later bags can
appear normally. Multiple bags have a tab switcher inside the core-centered modal.
Small visible core regions use a viewport-bounded modal so the result stays readable.
Both Chinese and English, dark and light themes, touch and keyboard are supported.

Countdowns use server-clock offset. WS state changes trigger private result refreshes;
reconnect refreshes recover missed draws. Only unresolved deadlines/results use a
five-second fallback poll. A closed modal still refreshes paid balance after winning.
Public broadcasts cannot erase private participation or display a premature loss.

## Verification

- `python -m unittest tests.test_live_lucky_bags`
- `node --test tests/liveLuckyBags.test.js` (from frontend)
- Local browser smoke script `output/live-lucky-check.js` mocks transactions and
  checks desktop/mobile, bilingual themes, entry, dismissal/reload, automatic balance
  refresh, and guest login. It creates no production bags or credits.
