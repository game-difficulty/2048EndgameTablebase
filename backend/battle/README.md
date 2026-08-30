# Battle mode architecture

Battle is split into a reusable room core and ruleset-specific modes.

## Reusable core

- `core/lifecycle.py`: membership, roles, ready state, lobby snapshots and room closure.
- `core/contracts.py`: the `BattleMode` contract.
- `core/registry.py`: backend mode registration and lookup.
- `realtime.py`: subscriptions, presence and the generic `BATTLE_ACTION` envelope.
- `routes.py`: common room HTTP endpoints and generic artifact downloads.
- `core/chat.py`: shared room chat history, validation, rate limiting and cleanup.

The core must not import a mode's scoring, artifact codec, tablebase provider or
board rules.

## Adding a mode

1. Add `modes/<mode_key>/` and implement `BattleMode`.
2. Register one mode instance during module import.
3. Keep generation, scoring, action validation and cleanup inside that mode.
4. Store durable mode configuration in `settings_json`, round state in
   `mode_state_json`, and result details in `mode_data_json`.
5. Add a frontend definition in `features/battle/core/modeRegistry.js` with its
   hall, match and result components plus a session adapter.

Existing relational goodness fields and `/route` remain compatibility surfaces.
New modes should use `/artifact` and `BATTLE_ACTION`.
