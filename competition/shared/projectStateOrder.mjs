// Uploads are complete states, so a receiver may safely skip intermediate
// sequences. Only an adjacent state can reuse the supplied slide animation.
export function receivedProjectView(previous, next) {
  if (!next) return previous;
  const generation = Number(next.generation || 0), oldGeneration = Number(previous?.generation || 0);
  if (previous && generation === oldGeneration && next.frames?.length) {
    const frames = new Map((previous.frames || []).map(frame => [frame.sequence, frame]));
    for (const frame of next.frames) frames.set(frame.sequence, frame);
    const latest = Number(next.sequence) >= Number(previous.sequence) ? next : previous;
    return { ...latest, frames: [...frames.values()].sort((a, b) => a.sequence - b.sequence).slice(-128) };
  }
  if (previous && (generation < oldGeneration
    || (generation === oldGeneration && Number(next.sequence) < Number(previous.sequence)))) return previous;
  if (!previous || generation !== oldGeneration || Number(next.sequence) > Number(previous.sequence) + 1) {
    return { ...next, payload: { ...next.payload, last_transition: { kind: 'restore' } } };
  }
  return next;
}

export function projectionIsOlder(previous, next) {
  if (!previous || (previous.match_public_key ?? previous.public_key) !== (next.match_public_key ?? next.public_key)) return false;
  const generation = Number(next.generation || 0), oldGeneration = Number(previous.generation || 0);
  return generation < oldGeneration || (generation === oldGeneration
    && Number(next.content_sequence) < Number(previous.content_sequence));
}
