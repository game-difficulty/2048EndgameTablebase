export const MIN_BC_FAMILY_MODULUS = 13;
export const MAX_BC_FAMILY_MODULUS = 256;
export const DEFAULT_BC_FAMILY_MODULUS = 29;

const isPrime = (value) => {
  if (value < 2) {
    return false;
  }
  if (value === 2) {
    return true;
  }
  if (value % 2 === 0) {
    return false;
  }

  for (let divisor = 3; divisor * divisor <= value; divisor += 2) {
    if (value % divisor === 0) {
      return false;
    }
  }
  return true;
};

export const normalizeBCFamilyModulus = (value) => {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    return DEFAULT_BC_FAMILY_MODULUS;
  }

  const clamped = Math.min(
    MAX_BC_FAMILY_MODULUS,
    Math.max(MIN_BC_FAMILY_MODULUS, parsed)
  );
  if (isPrime(clamped)) {
    return clamped;
  }

  const maxDistance = Math.max(
    clamped - MIN_BC_FAMILY_MODULUS,
    MAX_BC_FAMILY_MODULUS - clamped
  );
  for (let distance = 1; distance <= maxDistance; distance += 1) {
    const lower = clamped - distance;
    if (lower >= MIN_BC_FAMILY_MODULUS && isPrime(lower)) {
      return lower;
    }

    const upper = clamped + distance;
    if (upper <= MAX_BC_FAMILY_MODULUS && isPrime(upper)) {
      return upper;
    }
  }

  return DEFAULT_BC_FAMILY_MODULUS;
};
