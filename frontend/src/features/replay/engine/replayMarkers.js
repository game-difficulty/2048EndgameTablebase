export function normalizeReplayMarkerThreshold(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 1;
  return Math.min(1, Math.max(0, numeric));
}

function replayLossValues(losses) {
  return Array.from(losses || [], Number).filter(Number.isFinite);
}

function quantile(values, fraction) {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const ratio = position - lower;
  return sorted[lower] + (sorted[upper] - sorted[lower]) * ratio;
}

export function replayMarkerCutoff(losses, configuredThreshold = 1) {
  const values = replayLossValues(losses);
  const threshold = normalizeReplayMarkerThreshold(configuredThreshold);
  return values.length ? Math.min(quantile(values, 0.1), threshold) : threshold;
}

export function replayMarkerIndices(losses, configuredThreshold = 1) {
  const values = replayLossValues(losses);
  if (!values.length) return [];
  const cutoff = replayMarkerCutoff(values, configuredThreshold);
  return values
    .map((loss, index) => ({ loss, index }))
    .filter(({ loss }) => loss < 1 && loss < cutoff)
    .map(({ index }) => index);
}
