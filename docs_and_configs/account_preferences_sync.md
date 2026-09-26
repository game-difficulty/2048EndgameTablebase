# Account preference sync

`GET/PATCH /api/profile/preferences` stores a small allowlisted JSON document in
`auth.sqlite3.user_preferences`, keyed by account ID. Both main and play-site
frontends read it after authentication on page entry, and refresh it when their
settings view opens. Edits apply locally first, then send debounced field-level
PATCH requests. Failed edits remain in a per-account browser queue and the
settings view offers a retry. The server merges patches in a short transaction;
the initial browser migration uses `only_if_missing` so it cannot overwrite an
existing account preference.

Synced presentation fields: `language`, `dark_mode`, `theme`,
`use_custom_theme`, `custom_colors`, `font_size_factor`, `ui_scale`, and
`do_animation`. Synced play-site fields: `alwaysConfirmRestart`, `showSpeed`,
and `showFourPercent`. The four `display_thresholds` remain in
`human_player_settings`; each run snapshots its threshold at creation.

Touch swipe sensitivity, panel visibility, training controls, active games, and
replays stay browser-local. Resolved `colors` is derived from the theme or
custom palette and is not saved as an account field. The play site must derive
its palette from the account preference; its legacy palette cookie is only a
same-site presentation aid, not the source of truth.

`play.2048tables.online` is allowed to receive the existing parent-domain
session cookie. If the final play hostname differs, add its subdomain to
`AUTH_SHARED_COOKIE_HOSTS`. An unrelated registrable domain cannot share this
cookie; it must authenticate against the same account backend separately.
