// Uploads are complete states, so a receiver may safely skip intermediate
// sequences. Only an adjacent state can reuse the supplied slide animation.
export function receivedProjectView(previous, next) {
  if (!next) return previous;
  const generation = Number(next.generation || 0), oldGeneration = Number(previous?.generation || 0);
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
