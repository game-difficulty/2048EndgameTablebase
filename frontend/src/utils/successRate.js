const ONE_MINUS_SUCCESS_RATE_DTYPES = new Set(['1-float32', '1-float64']);
const ONE_MINUS_TEXT_THRESHOLD = -1e-7;
const FLOAT_SUCCESS_RATE_DTYPES = new Set(['float32', 'float64']);
const SCIENTIFIC_TEXT_THRESHOLD = 1e-6;

const isFiniteNumber = (value) => typeof value === 'number' && Number.isFinite(value);
const trimTrailingZeros = (value) => value.replace(/(\.\d*?[1-9])0+$/u, '$1').replace(/\.0+$/u, '').replace(/\.$/u, '');
const trimScientificZeros = (value) => value.replace(/(\.\d*?[1-9])0+(e[+-]?\d+)$/iu, '$1$2').replace(/\.0+(e[+-]?\d+)$/iu, '$1');

export const isOneMinusSuccessRateDtype = (dtype) => ONE_MINUS_SUCCESS_RATE_DTYPES.has(dtype);

export const restoreSuccessRate = (value, dtype) => {
  if (!isFiniteNumber(value)) return null;
  if (!isOneMinusSuccessRateDtype(dtype)) return value;
  return 1 + value;
};

export const successRateSortValue = (value, dtype) => {
  if (!isFiniteNumber(value)) return null;
  if (isOneMinusSuccessRateDtype(dtype)) return value;
  return restoreSuccessRate(value, dtype);
};

export const successRateRelativeLoss = (value, bestValue, dtype) => {
  if (!isFiniteNumber(value) || !isFiniteNumber(bestValue)) return null;
  if (isOneMinusSuccessRateDtype(dtype)) {
    return Math.max(0, bestValue - value);
  }
  const restored = restoreSuccessRate(value, dtype);
  const bestRestored = restoreSuccessRate(bestValue, dtype);
  if (!isFiniteNumber(restored) || !isFiniteNumber(bestRestored) || bestRestored <= 0) return null;
  return 1 - restored / bestRestored;
};

export const formatSuccessRate = (value, dtype, precision = 9) => {
  if (!isFiniteNumber(value)) return String(value);

  if (!isOneMinusSuccessRateDtype(dtype)) {
    if (
      FLOAT_SUCCESS_RATE_DTYPES.has(dtype) &&
      value !== 0 &&
      Math.abs(value) < SCIENTIFIC_TEXT_THRESHOLD
    ) {
      return trimScientificZeros(value.toExponential(Math.max(precision, 12)));
    }
    return trimTrailingZeros(value.toFixed(precision));
  }

  if (value >= 0 || value < ONE_MINUS_TEXT_THRESHOLD) {
    return trimTrailingZeros((1 + value).toFixed(precision));
  }

  return `1${String(value)}`;
};

export const resultValueFontSize = (displays) => {
  const maxLength = Math.max(0, ...displays.map((display) => String(display || '').length));
  if (maxLength >= 21) return '15px';
  if (maxLength >= 19) return '16px';
  if (maxLength >= 17) return '17px';
  return '19px';
};
