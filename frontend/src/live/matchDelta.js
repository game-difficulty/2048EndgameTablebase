// This baseline belongs only to one board socket; HTTP recovery cannot replace it.
function patch(base, change) {
  if (!change || typeof change.set !== 'object' || !Array.isArray(change.unset)) throw Error('invalid_delta');
  const result = { ...base, ...change.set };
  for (const key of change.unset) delete result[key];
  return result;
}

function unpackView(view) {
  if (!view?.payload_from_last_frame) return view;
  const { payload_from_last_frame: _, ...rest } = view;
  const last = rest.frames?.at(-1);
  if (!last || last.sequence !== rest.sequence || !Object.hasOwn(last, 'payload')) throw Error('invalid_view_delta');
  return { ...rest, payload: last.payload };
}

export function createMatchDeltaDecoder(room) {
  let baseline = null, epoch = null, sequence = null;
  function reset() { baseline = null; epoch = sequence = null; }
  function decode(data) {
    if (data.type !== 'snapshot' && data.type !== 'match_delta') return data;
    if (data.room_id !== room.id || data.protocol !== room.protocol) throw Error('wrong_delta_room');
    if (data.type === 'snapshot') {
      if (typeof data.stream_epoch !== 'string' || !Number.isSafeInteger(data.stream_sequence)) throw Error('missing_delta_version');
      baseline = data; epoch = data.stream_epoch; sequence = data.stream_sequence;
      return data;
    }
    if (!baseline?.match || data.stream_epoch !== epoch || data.base_sequence !== sequence
        || !Number.isSafeInteger(data.stream_sequence) || data.stream_sequence !== sequence + 1) throw Error('delta_gap');
    const views = patch(baseline.match.project_public_views || {}, data.views);
    const match = { ...patch(baseline.match, data.match), project_public_views: Object.fromEntries(
      Object.entries(views).map(([side, view]) => [side, unpackView(view)])) };
    baseline = { ...patch(baseline, data.fields), type: 'snapshot', match,
      stream_epoch: epoch, stream_sequence: data.stream_sequence };
    sequence = data.stream_sequence;
    return baseline;
  }
  return { decode, reset };
}
