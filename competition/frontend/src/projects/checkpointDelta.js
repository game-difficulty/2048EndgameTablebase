const copy = value => JSON.parse(JSON.stringify(value));
const lists = checkpoint => ({ metric_history: checkpoint.metric_history || [],
  undo: checkpoint.state.undo || [], lookBackHistory: checkpoint.state.lookBackHistory || [] });
export function checkpointDelta(previous, next, baseSequence) {
  const result = copy(next), before = lists(previous), after = lists(next);
  delete result.metric_history; delete result.state.undo; delete result.state.lookBackHistory;
  const patches = {};
  for (const key of Object.keys(after)) {
    let keep = 0;
    while (keep < before[key].length && keep < after[key].length
      && JSON.stringify(before[key][keep]) === JSON.stringify(after[key][keep])) keep++;
    patches[key] = { keep, append: after[key].slice(keep) };
  }
  return { ...result, delta_base: baseSequence, lists: patches };
}
