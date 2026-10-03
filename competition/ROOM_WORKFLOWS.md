# Reusable room workflows

## Boundaries

- `backend/room_rules.py`: versioned, event-independent configuration validation, lineup policies and series termination. No game/project imports.
- `backend/room_draft.py`: persistent ordered pick/ban state machine. Project keys are opaque; availability is taken from the room's frozen pool.
- `backend/service.py`: authoritative room orchestration, identity/permissions, clocks, adapter sessions, readiness, results and public projections. Both old and configurable rooms use the same match runtime.
- `backend/event_catalog.py`, `event_enrollment.py`, `event_schedule.py`: optional event membership, roster binding and scheduled attendance. Standalone rooms do not require an event.
- `shared/DraftWorkflow.vue`: shared participant/spectator presentation; only participant callers enable command controls.

## Creating a room

`GET /api/room-rule-presets` returns the authoritative presets. Pass the chosen editable fields as `rules` to `POST /api/competitions`:

```json
{"preset":"bo7","team_size":5,"series_mode":"all","lineup_policy":"balanced","final_selection":"random","draft_seconds":60,"lineup_seconds":180,"team_clock_seconds":1800}
```

Presets:

| Template | Ordered steps | Last game | Minimum pool |
| --- | --- | --- | --- |
| BO3 | First B1P1 → Second B1P1 | Blind picks, then draw | 5 |
| BO5 | First B1P1 → Second B1P2 → First P1 | Remaining-pool draw | 7 |
| BO7 | First B1P1 → Second B2P2 → First B1P2 → Second P1 | Remaining-pool draw | 11 |

BO3 defaults to all three games, unique players, and three seats per team. BO5/BO7 default to first-to-majority and repeated appearances; every choice is explicit and editable before creation. Games may draw: draws do not count as wins, and the series always ends at its configured game limit. `all` never ends early. Match points remain 2 per game win / 1 per draw.

`custom` additionally accepts `steps: [{actor: "first"|"second", bans: N, picks: N}]`. There are 1–24 nonempty steps, 1–16 players per side, and 1–15 odd-numbered games including one final draw. Pools contain 1–32 registered project snapshots and must cover all bans plus games. First/second refer to the initial side draw, not fixed colors. Picks and bans in one step are submitted together; pick list order determines game order.

Lineup policies: `unique` means at most once per player; `everyone` allows repetition but requires the maximum possible number of distinct players, `min(team_size, game_count)`; `balanced` means appearance counts across the whole roster (including zero) differ by at most one; `free` allows unrestricted repetition. For three players and seven games, `everyone` permits 5/1/1 whereas `balanced` requires a permutation of 3/2/2. New BO5/BO7 presets default to `everyone`; existing frozen room rules remain unchanged. Policies validate the full planned lineup, not the prefix actually played if the series ends early. Use `series_mode: all` when the full allocation must be played. Timeouts assign seats cyclically in registration order, satisfying every supported policy. Seat 1 remains captain.

Per-step BP and lineup timers accept 5–3600 seconds; team clocks accept 30–86400 seconds. Existing ready-check, late-forfeit and result-rest policies remain unchanged. A late team forfeits all configured games, not a hard-coded three. Both absent remains 0:0.

## State / commands

New configured rooms persist normalized immutable rules in `competition_room_rules`. Ordered selections, bans and audit history are persisted in `competition_draft_steps`. `DRAFT_STEP` repeats with a new phase token for each step. `POST /api/competitions/{code}/draft/step` accepts `picks`, `bans`, `phase_token`, `command_id`; only the active captain may submit. Duplicate commands are idempotent and stale tokens are rejected. On timeout, picks then bans take the first eligible pool entries, preserving legacy fallback behavior.

Private snapshots and public live projections include `rules` and `draft.workflow` / `public_draft.workflow`. The latter contains steps, current actor, chosen projects and history, never the opponent's unsubmitted blind pick or private lineup. The draft seed is not revealed before the match finishes because game seeds are derived from it. Game keys extend from A through O. `C_DRAW` is retained as the wire-level final-draw/reveal phase name for compatibility; it does not mean the series has only three games.

## Compatibility and release

- Omitted `rules` retains the legacy BO3 draft path. Existing rooms receive no new configuration row; their old seating, draft, lineup and completion semantics are preserved.
- SQLite schema 16 widens game/seat constraints and removes the one-game-per-seat lineup uniqueness constraint. Migration copies existing rows transactionally, recreates explicit indexes and verifies foreign keys. Application validation preserves uniqueness in legacy rooms.
- New events can specify `team_size`; existing event roster sizes are not modified. Bound rooms must match the locked event roster size.
- Legacy betting is offered only for three-game, play-all rooms. Other formats return no prediction window; no BO3 markets should be created for them.
- Release the tournament and live readers together with the backend capability. Do not expose configurable room creation to readers that only understand A/B/C. Follow the repository's production backup/retention and verified-source deployment process. This implementation does not deploy automatically.

Verification covers legacy tests, custom validation, BO5/BO7 drafts, private blind choices, repeated lineups, real A–G sessions, timeout fallback, idempotency, early termination, roster sizes, migration and API authorization. UI screenshots are local development artifacts under `output/playwright/`.

## Free duels / fixed sequences

`backend/room_flow.py` is the small common boundary for a match plan, readiness
holds and starting a prepared room. Draft rooms continue through the existing BP
engine. A fixed sequence stores its ordered project keys and private seed in
`competition_fixed_series`, without fabricating a draft or secret lineup. Both
paths share sessions, clocks, result publication, phase tokens, recovery and score
aggregation. Project descriptors declare a `ResultPolicy`; the room service no
longer infers a scoring metric from a renderer's protocol. Old descriptor wire
snapshots remain compatible and rules versions still pin adapter selection.

`backend/duel_rooms.py` provides the separate public-room creation policy. The
authenticated `POST /api/duel-rooms` accepts `name`, `projects` (ordered unique
project references, 1–15, including even counts), `clock_seconds` (30–86400,
default 1800) and `command_id`. Unknown extra fields are rejected. The public
`GET /api/duel-projects` returns registered non-test projects; the last registered
version of each project is offered for new rooms. The server, not the caller,
chooses and freezes the name, adapter and version. Old versions remain available
to rooms already using them.

The host takes Yellow seat 1 and cannot hand off that seat; another account joins
White. Both lobby-ready commands start Game A directly. Later games require one
explicit readiness command per player (using the existing player-readiness
endpoint). No preview, prediction wait, BP, secret lineup or timeout auto-ready
is inserted. The existing 30-second result display and shared per-side match
clock, including Higher refunds and clock-expiry settlement, are retained. All
selected games are played unless the existing clock-expiry policy ends play;
ties are allowed. Scores count game wins, not sums of unlike project metrics.

No organizer/referee role is granted, and even platform officials cannot invoke
staff assignment, member removal, rematching, suspension, result override or
force-finish on these rooms. Hosts can close their own room before play. Duels
cannot be linked to events, produce no official record eligibility, do not
create prediction markets and are not published in the official live directory.

Abuse/lifecycle defaults: at most one active duel per account (host or seated
opponent), one creation per 60 seconds and at most ten per rolling hour. Creation
and joining checks use immediate transactions; repeated creation with the same
command ID returns the original room. Pre-start and between-game waiting expire
after 30 minutes; reads/readiness toggles do not extend the deadline. Expiration
closes without assigning a winner and preserves completed results. Schema 17
adds only the two new tables and their index; existing rooms are not rewritten.

The `/duels` frontend uses separate creation/waiting components, supports ordered
selection on touch devices and English/Chinese text, and reuses the existing
room URL, transport, gameplay and settlement components. Snapshots expose
`room_kind`, fixed `selected_projects` and `waiting_expires_at`; they do not expose
the private series seed or leak legacy blind choices.
# Timed personal-best rooms (schema 18)

`/time-attacks` is a separate public 1v1 mode. `time_attack_rooms.py` owns the
fixed-window attempt lifecycle, not the existing Faster/first-to-target logic.
`projects/target_challenge.py` is its server-owned, versioned parameter registry
and target/result contract. It reuses the standard deterministic human-play
engine, without importing account history, leaderboard or reward services.

- Four boards: 4x4, 3x4, 2x4 and 3x3; standard 90% 2 / 10% 4 spawns.
- Tile targets: powers of two from 8 through 2^31; reaching a larger tile counts.
- Board-sum targets: even integers from 10 through 2^31, exact equality after
  the move's spawn. Overshooting ends that attempt without a result.
- Duration: 30–86400 seconds, frozen with the project version and target at creation.
- Both players explicitly ready up. The server creates their first attempts
  together and persists a common deadline. No BP, lineup, referee or auto-ready.
- Unlimited attempts during the window, with a 500 ms restart anti-spam interval;
  no undo. Each attempt receives a fresh private random seed. Death or reaching
  the target stops only that attempt. Restart preserves all previous valid PBs.
- An attempt starts when its board is created on the server, not on first input.
  **Official PB is server-confirmed elapsed milliseconds**, including delivery
  latency. Client stopwatch values are explanatory, not trusted result values.
  Moves are admitted inside the serialized transaction before the deadline;
  at or after the deadline no new attempt or move is accepted. No late-upload grace.
  Refresh, disconnect, background tabs and server restarts do not pause the window.
- Lowest valid PB wins. A sole valid finisher wins; equal PBs or neither finishing
  are draws. Incomplete attempts at the deadline do not count. Finalization is
  idempotent and runs both from the normal deadline worker and request catch-up.

Shared infrastructure: `DuelRooms.create` has an internal, server-validated
challenge path; it persists the same public-room ownership/expiry metadata.
Both modes share one active room per user, a 60-second create cooldown, a limit
of 10 creations/hour and a 30-minute pre-start expiry. `RoomFlow.begin` dispatches
to the mode. Existing seat/ready/close/room-snapshot/WebSocket interfaces remain.
All official/event-linking/forced-result permissions remain denied, including to
platform admins; no official live listing, predictions, event points or rewards.
The target descriptor is deliberately kept out of the fixed-project duel picker:
it requires this parameterized workflow, not a fake ordinary-game adapter.

Additive tables: `competition_time_attack` (frozen config, start, deadline, winner)
and `competition_time_attempts` (side, sequence number, seed, verified state,
compact operation record, status, PB). Old room tables and semantics are retained.

New APIs:

- `POST /api/time-attack-rooms`: name, variant, target_kind (`tile`/`board_sum`),
  target_value, clock_seconds, command_id. Extra configuration/privilege keys rejected.
- `POST /api/competitions/{code}/time-attack/attempt`: action (`submit`/`restart`),
  attempt_id, command_id; submit adds base_sequence and up to 64 `[event_code,delta_ms]`
  pairs. The server replays and verifies movement and seeded spawns; it computes
  the first target hit itself, rejects post-completion moves, and never trusts
  supplied boards, completion flags or PBs. Command retries are idempotent; stale
  attempt IDs or sequence numbers cannot change a newer attempt.
- `GET /api/competitions/{code}/time-attack/best/{side}`: verified best-run replay.
  Only completed PB seeds can be read by other users; active RNG/state is private.

Room snapshots add `room_kind=time_attack` and `time_attack` configuration, timing,
per-side current attempt, PB reference and completion count. The ordinary `match`
payload stays null: attempts are not synthetic games or draft rounds.
The bilingual UI uses an optimistic deterministic board plus a serialized,
retryable sender; uncertain writes retry the same command ID. It resumes the last
server-confirmed state after reload (unacknowledged local moves may be lost).
PB replay uses the same verified operation codec; future tournament integration
can reuse the target contract and timed-series controller without adding privileges
to these public rooms.
