// Absolute move instants. Event deltas remain the compact on-wire representation.
export function recordMoveTime(run, stamp) {
  const raw = run.seq ? stamp - run.lastActionAt : 0;
  const delta = Math.max(0, Math.min(0xfffffffe, raw));
  const prior = run.wallTimeline;
  const anchors = (prior?.anchors || []).map(item => [...item]);
  if (!anchors.length || raw !== delta) anchors.push([run.seq + 1, stamp]);
  // Bound the HTTP metadata even on a machine whose wall clock keeps changing.
  const truncatedAt = prior?.truncated_at_seq || (anchors.length > 64 ? run.seq + 1 : null);
  return { delta, timeline: { version: 1, anchors: anchors.slice(0, 64),
    started_at_ms: prior?.started_at_ms ?? null, truncated_at_seq: truncatedAt } };
}

export function moveInstants(timeline, events) {
  const result = Array(events.length).fill(null);
  if (timeline?.version !== 1 || !Array.isArray(timeline.anchors)) return result;
  const anchors = new Map(timeline.anchors);
  let stamp = null;
  events.forEach((event, i) => {
    stamp = anchors.has(i + 1) ? anchors.get(i + 1) : stamp === null ? null : stamp + event[1];
    if (!timeline.truncated_at_seq || i + 1 < timeline.truncated_at_seq) result[i] = stamp;
  });
  return result;
}
