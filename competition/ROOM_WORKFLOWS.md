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
