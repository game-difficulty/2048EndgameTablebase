# Gamer AI continuous route stream

## Scope

The Gamer table AI uses a reusable authenticated browser WebSocket. A route is
opened once and advanced by a deterministic cursor. Results are delivered one
node at a time. It no longer requests a new four-node HTTP batch when a cache
window runs out. The old HTTP endpoint remains available for older clients.

The browser still owns movement, animation, RNG and the final choice between
tablebase candidates and search AI. Structural candidate filtering is unchanged.
Predictions stop before an EvilGen spawn, which the browser computes locally;
the next actual position starts a new route. Changing patterns, rates, difficulty,
board, account, or stopping AI cancels outstanding speculation.

## Flow control

- `GAMER_STREAM_OPEN`: original route request (`steps: 1`), `route_id`, `received`,
  `resume`, and initial `allow_through: 7`.
- `GAMER_STREAM_EVENT`: `result` frames with consecutive zero-based `seq`, exact
  board/RNG, results, catalog version and token balance; or `window`, `end`, `error`.
- `GAMER_STREAM_CREDIT`: cumulative `consumed` and `allow_through` indices. Credits
  allow future work, not acknowledgement-gated moves. Cached moves never await
  the `window` response; it only measures RTT for the adaptive window.
- `GAMER_STREAM_CANCEL`: stops a subscription without closing the reusable socket.

The initial window is eight nodes. It adapts to observed RTT and consumption
interval, up to 32 nodes ahead. Credits are coalesced, with a refill margin that
includes their four-move cadence. Paid results remain in a 256-entry in-memory
LRU with a five-minute TTL. Nothing is saved to localStorage.

The Worker advertises `gamer_stream_v1` and handles OPEN/CREDIT/CANCEL on its
existing connection. Each lookup immediately emits `GAMER_STREAM_NODE`; the
route ends with `GAMER_STREAM_END`. It does not buffer the whole route. Native
reader resource/table semaphores are released between reads, and readers are
reused. Local tables use the existing query scheduler/cache. Old Workers fall
back to ordinary per-node lookup, preserving functionality but not eliminating
the cloud-to-Worker RTT; upgrading the Worker is required for the full benefit.

## Safety and billing

Pricing and hit/miss semantics are unchanged. Delivered prefetch nodes are paid
queries, as before; a wider window can therefore purchase more unused nodes
when AI is stopped. Work is bounded by available balance and the granted window.
Charges retain the `gamer-ai:{request_id}:{seq}` idempotency key and bind it to
the request fingerprint. No new database table or per-step diagnostic log.

Each user has at most two active subscriptions. Global admission is capped at
24 for ordinary users and 32 for supporters. This supplements existing socket
limits and scheduler priority; it does not increase native reader concurrency.
Sessions and catalog versions are rechecked during generation.

Disconnects retain bounded paid frames in memory for up to 90 seconds of idle
time. Resume requires the same authenticated session and identical request.
Only missing frames are resent, without another lookup or charge. After process
restart or retention expiry, `STREAM_GONE` ends the old subscription; the browser
keeps its existing cache and opens a new route when needed. Resume is not durable
across a server restart. Cancel stops queued work; an already-running native
read or atomic billing transaction must finish before its resources are released.

## Performance expectations

The first lookup still pays initial network latency. After warm-up, generation,
transmission and playback overlap rather than alternate in four-node batches.
This hides latency when the buffer covers RTT and the producer keeps up. It
cannot guarantee zero stalls during long disconnects, an EvilGen boundary, a
pattern switch, or consumption faster than the bounded window/producer allows.

## Verification and rollout

- `npm run test:gamer-table-ai`: sustained consumption, simulated 300ms RTT,
  cancellation, reconnect receipts, LRU/TTL, structural filtering.
- Python `test_gamer_table_stream.py`: credits, authentication, quota, resend
  without recharging and the actual app send adapter.
- Worker `test_client_protocol.py`: node zero arrives while lookup one is still
  blocked, credit ceiling, continuation and cancellation.
- `test_remote_tablebase_workers.py`: ordered Worker node delivery and end frames.
- Build frontend and compile backend/Worker; test actual browser plus a local
  Worker against a real table before rollout.

Deploy cloud code/assets and update/restart the local Worker. No native rebuild
or schema migration is required. Retain the old HTTP endpoint during rollout so
already-open older browser clients can continue to work.
