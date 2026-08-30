import {
  TRAINER_ROUTE_DIRECTIONS,
  TRAINER_ROUTE_RATE_SCALE,
} from './battleRouteCodec.js';

export const TESTER_PERFECT_EPSILON = 3e-10;

export function normalizeBattleDirection(direction) {
  const normalized = String(direction || '').trim().toLowerCase();
  return TRAINER_ROUTE_DIRECTIONS.includes(normalized) ? normalized : null;
}

export function directionRate(rates, direction) {
  const normalized = normalizeBattleDirection(direction);
  const index = normalized == null ? -1 : TRAINER_ROUTE_DIRECTIONS.indexOf(normalized);
  if (index < 0 || rates == null || rates.length <= index) return null;
  const value = Number(rates[index]);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

export function bestRateForBattleStep(rates) {
  if (rates == null || rates.length < 4) return null;
  let bestRate = -1;
  let bestDirection = null;
  TRAINER_ROUTE_DIRECTIONS.forEach((direction, index) => {
    const value = Number(rates[index]);
    if (Number.isFinite(value) && value > bestRate) {
      bestRate = value;
      bestDirection = direction;
    }
  });
  return bestDirection == null ? null : { bestDirection, bestRate };
}

export function scoreBattleStep({
  rates,
  selectedDirection,
  standardDirection = null,
  goodnessOfFit = 1,
} = {}) {
  const selected = normalizeBattleDirection(selectedDirection);
  const standard = normalizeBattleDirection(standardDirection);
  const best = bestRateForBattleStep(rates);
  const selectedRate = directionRate(rates, selected);
  if (!best || selected == null || selectedRate == null) {
    throw new TypeError('battle_step_scoring_data_required');
  }

  const bestDirection = standard || best.bestDirection;
  const standardRate = directionRate(rates, bestDirection);
  if (standardRate == null) throw new TypeError('battle_standard_direction_required');
  const rateTolerance = TESTER_PERFECT_EPSILON * TRAINER_ROUTE_RATE_SCALE;
  if (best.bestRate - standardRate > rateTolerance) {
    throw new RangeError('battle_standard_direction_not_optimal');
  }

  const selectedNormalized = selectedRate / TRAINER_ROUTE_RATE_SCALE;
  const bestNormalized = best.bestRate / TRAINER_ROUTE_RATE_SCALE;
  const ratio = Math.abs(bestNormalized - selectedNormalized) <= TESTER_PERFECT_EPSILON
    ? 1
    : (bestNormalized > 0 ? selectedNormalized / bestNormalized : 1);
  const previous = Number.isFinite(Number(goodnessOfFit))
    ? Math.max(0, Number(goodnessOfFit))
    : 1;
  const next = previous * ratio;

  return {
    selectedDirection: selected,
    bestDirection,
    selectedRate,
    bestRate: best.bestRate,
    ratio,
    stepLoss: 1 - ratio,
    goodnessDrop: 1 - ratio,
    goodnessOfFitBefore: previous,
    goodnessOfFit: next,
    goodnessOfFitDrop: previous - next,
    perfect: ratio > 1 - TESTER_PERFECT_EPSILON,
  };
}
