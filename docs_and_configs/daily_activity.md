# Daily active accounts

The admin chart uses Beijing calendar days. An account counts once per day
across the main site, Play and Live if a recorded visit or a user-initiated
operation exists. Anonymous visitors and guests without an account are excluded.

`daily_user_activity(day, user_id, site)` is the durable source. The shared
`/api/auth/me` and successful authentication record main, Play and Live visits.
The Live watch socket also records authenticated viewing when it crosses
midnight. The idempotent backfill script also reads login sessions, quota
operations, ranked starts, minigame starts, battle membership/chat, Live gift
and draw participation from the auth database. It reads Play run starts and
received move batches from `HUMAN_PLAY_DB`. Automated grants, awards and
settlements are excluded. A user active on multiple sites still counts once.

After installing the new backend, backfill the chart's history once:

```sh
cd /opt/2048tables/app
/opt/2048tables/venv/bin/python -m scripts.refresh_daily_activity --days 14 --play-db /var/lib/2048tables/play/human.sqlite3
```

Install and enable `deploy/2048tables-daily-activity.service` and its timer to
fill missed activity for the chart's 14-day window every day at
00:10 Beijing time. The units load the Cloud and optional Play environment
files and pass the production Play database path explicitly.

Older page views and Live viewers without recorded actions cannot be
reconstructed. Thus pre-deployment values are lower bounds; compare trends
across the instrumentation rollout with care.
