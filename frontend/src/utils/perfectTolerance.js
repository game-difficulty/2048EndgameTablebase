import config from '../../../docs_and_configs/performance_evaluations.json' with { type: 'json' };

export function perfectTolerance(dtype = 'uint32') {
  const name = String(dtype || '').trim().toLowerCase();
  const aliases = { f32: 'float32', f64: 'float64', '1-f32': '1-float32', '1-f64': '1-float64' };
  const perfect = config.perfect;
  return Number(perfect.tolerance_by_dtype[aliases[name] || name] ?? perfect.tolerance);
}

export function isPerfectResult(selectedRate, bestRate, dtype = 'uint32') {
  if (selectedRate == null || bestRate == null) return false;
  const selected = Number(selectedRate);
  const best = Number(bestRate);
  return Number.isFinite(selected) && Number.isFinite(best)
    && best - selected <= perfectTolerance(dtype);
}
