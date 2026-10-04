# Play live bandwidth improvements

Ordinary moves remain 5-byte publisher events and 9-byte viewer events. These
changes affect bootstrapping and metadata, not authoritative Play uploads.

## Publisher recovery

The signed lease descriptor advertises `resume_supported: true`. A supporting
client includes `resume: true` in `hello`/`switch`, then waits for
`prefix_request {run_id, start, seq}` before sending an HLP1 packet. Only events
in `[start, seq)` are uploaded. Steps created during negotiation wait for `ready`
and are then streamed normally.

The server chooses its retained sequence only if run ID, variant and seed match
and the retained sequence is no greater than the proposed sequence. Otherwise it
requests start 0. The HLP1 compression/layout rules remain unchanged. Recovery
validates and reconstructs a separate state before replacing the retained run;
failed validation leaves the retained state intact. Same-run recovery preserves
verified milestone bookkeeping. New-run recovery resets it. Viewers receive one
current snapshot after recovery, not the entire replay prefix.

Old descriptors cause the client to send the legacy full prefix immediately.
The server still accepts legacy publishers without `resume`. Deploy the live
receiver before publishing the Play lease capability and the new Play client.
During rollback, disable the advertised capability before reverting the receiver.

## Viewer bootstrap

Human rooms obtain the current board from the initial WebSocket snapshot. They
do not request `/state` first. `/social-state` supplies chat/gift history and
music without repeating the board or AI statistics. Socket-open and foreground
requests share one in-flight fetch and reuse a successful result for 2 seconds;
failed requests remain immediately retriable. AI/competition bootstrapping is
unchanged by these optimizations.

## Best score

Current-game best is derived from the streamed score on both receiver and viewer.
The publisher sends `best` only when the external best exceeds both current score
and its last announced best. Hello still includes the initial external best, and
ready rechecks it to cover changes made during negotiation.

Validation: frontend humanLiveSwitch/sharedRefresh tests, backend human live and
resume tests. No database migration or replay format change.
