export const DEFAULT_PERFORMANCE_CONFIG = {
  report_decimal_places: 4,
  perfect: {
    label: 'Perfect!',
    comparison: 'absolute_difference',
    tolerance: 3e-10,
    color: '#2e7d32',
  },
  evaluations: [
    { label: 'Excellent!', threshold: 0.999, color: '#7cb342' },
    { label: 'Nice try!', threshold: 0.99, color: '#c0ca33' },
    { label: 'Not bad!', threshold: 0.975, color: '#fb8c00' },
    { label: 'Mistake!', threshold: 0.9, color: '#f4511e' },
    { label: 'Blunder!', threshold: 0.75, color: '#e53935' },
    { label: 'Terrible!', threshold: -1, color: '#b71c1c' },
  ],
  result_bar: {
    stops: [
      { max_loss: 0.001, color: '#2e7d32' },
      { max_loss: 0.01, color: '#8bc34a' },
      { max_loss: 0.03, color: '#ff9800' },
      { max_loss: 0.1, color: '#f44336' },
    ],
  },
  analysis: {
    start_step: 5,
    report_min_text_lines: 101,
  },
};

const cloneDefaults = () => ({
  ...DEFAULT_PERFORMANCE_CONFIG,
  perfect: { ...DEFAULT_PERFORMANCE_CONFIG.perfect },
  evaluations: DEFAULT_PERFORMANCE_CONFIG.evaluations.map((item) => ({ ...item })),
  result_bar: {
    stops: DEFAULT_PERFORMANCE_CONFIG.result_bar.stops.map((item) => ({ ...item })),
  },
  analysis: { ...DEFAULT_PERFORMANCE_CONFIG.analysis },
});

const finiteNumber = (value, fallback) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

export const normalizePerformanceConfig = (value = {}) => {
  const fallback = cloneDefaults();
  const perfect = value?.perfect || {};
  const evaluations = Array.isArray(value?.evaluations)
    ? value.evaluations
      .map((item, index) => ({
        label: String(item?.label || '').trim(),
        threshold: finiteNumber(item?.threshold, Number.NaN),
        color: String(item?.color || fallback.evaluations[index]?.color || 'var(--accent)').trim(),
      }))
      .filter((item) => item.label && Number.isFinite(item.threshold))
      .sort((a, b) => b.threshold - a.threshold)
    : [];
  const stops = Array.isArray(value?.result_bar?.stops)
    ? value.result_bar.stops
      .map((item, index) => ({
        max_loss: finiteNumber(item?.max_loss, Number.NaN),
        color: String(item?.color || fallback.result_bar.stops[index]?.color || '#f44336').trim(),
      }))
      .filter((item) => Number.isFinite(item.max_loss) && item.max_loss >= 0 && item.max_loss <= 1)
      .sort((a, b) => a.max_loss - b.max_loss)
    : [];

  return {
    report_decimal_places: Math.min(15, Math.max(0, Math.trunc(
      finiteNumber(value?.report_decimal_places, fallback.report_decimal_places)
    ))),
    perfect: {
      label: String(perfect.label || fallback.perfect.label).trim() || fallback.perfect.label,
      comparison: perfect.comparison === 'relative_ratio' ? 'relative_ratio' : 'absolute_difference',
      tolerance: Math.max(0, finiteNumber(perfect.tolerance, fallback.perfect.tolerance)),
      color: String(perfect.color || fallback.perfect.color).trim() || fallback.perfect.color,
    },
    evaluations: evaluations.length ? evaluations : fallback.evaluations,
    result_bar: { stops: stops.length ? stops : fallback.result_bar.stops },
    analysis: {
      start_step: Math.max(0, Math.trunc(finiteNumber(
        value?.analysis?.start_step,
        fallback.analysis.start_step
      ))),
      report_min_text_lines: Math.max(0, Math.trunc(finiteNumber(
        value?.analysis?.report_min_text_lines,
        fallback.analysis.report_min_text_lines
      ))),
    },
  };
};

export const performanceLevels = (config) => {
  const normalized = normalizePerformanceConfig(config);
  return [normalized.perfect, ...normalized.evaluations];
};

export const evaluationColor = (label, config) => (
  performanceLevels(config).find((item) => item.label === label)?.color || 'var(--accent)'
);

const parseHexColor = (color) => {
  const match = /^#([0-9a-f]{6})$/iu.exec(String(color || '').trim());
  if (!match) return null;
  return [0, 2, 4].map((offset) => Number.parseInt(match[1].slice(offset, offset + 2), 16));
};

const mixColors = (start, end, ratio) => {
  const first = parseHexColor(start);
  const second = parseHexColor(end);
  if (!first || !second) return ratio < 0.5 ? start : end;
  const mixed = first.map((channel, index) => (
    Math.round(channel + (second[index] - channel) * ratio)
  ));
  return `rgb(${mixed[0]}, ${mixed[1]}, ${mixed[2]})`;
};

export const resultBarPresentation = (relativeLoss, isBest, config) => {
  const stops = normalizePerformanceConfig(config).result_bar.stops;
  const first = stops[0];
  const last = stops[stops.length - 1];
  if (isBest) return { pct: 100, color: first.color };

  const loss = Number(relativeLoss);
  if (!Number.isFinite(loss) || loss > last.max_loss) {
    return { pct: 0, color: last.color };
  }
  const pct = last.max_loss > 0 ? Math.max(0, (1 - loss / last.max_loss) * 100) : 0;
  if (loss <= first.max_loss) return { pct, color: first.color };

  for (let index = 1; index < stops.length; index += 1) {
    const previous = stops[index - 1];
    const current = stops[index];
    if (loss <= current.max_loss) {
      const span = current.max_loss - previous.max_loss;
      const ratio = span > 0 ? (loss - previous.max_loss) / span : 1;
      return { pct, color: mixColors(previous.color, current.color, ratio) };
    }
  }
  return { pct: 0, color: last.color };
};

export const evaluationColorForRatio = (ratio, config) => {
  const normalized = normalizePerformanceConfig(config);
  const numeric = Number(ratio);
  if (numeric >= 1 - normalized.perfect.tolerance) return normalized.perfect.color;
  return normalized.evaluations.find((item) => numeric >= item.threshold)?.color
    || normalized.evaluations.at(-1)?.color
    || normalized.perfect.color;
};
