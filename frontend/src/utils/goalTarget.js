import { restoreSuccessRate } from './successRate.js';

export function parseGoalTarget(target) {
  const token = String(target ?? '');
  const sum = /^sum-(\d+)$/u.exec(token);
  const value = Number(sum ? sum[1] : token);
  if (sum && value >= 4 && value < 16384 && value % 2 === 0) return { kind: 'sum', value, token };
  if (!sum && /^\d+$/u.test(token) && value >= 2 && value <= 16384 && (value & (value - 1)) === 0) return { kind: 'tile', value, token };
  return null;
}
export const isSumTarget = target => parseGoalTarget(target)?.kind === 'sum';
export function compareGoalTargets(a, b) {
  const left = parseGoalTarget(a), right = parseGoalTarget(b);
  if (!left || !right) return String(a).localeCompare(String(b));
  return Number(left.kind === 'sum') - Number(right.kind === 'sum') || left.value - right.value;
}
export function goalTargetLabel(target, language = 'en') {
  const goal = parseGoalTarget(target);
  return goal?.kind === 'sum' ? `${String(language).startsWith('zh') ? '盘面和' : 'Board sum'} ${goal.value}` : String(target || '');
}
export function fullPatternLabel(value, language = 'en') {
  const text = String(value || ''), split = text.lastIndexOf('_');
  return split < 0 ? text : `${text.slice(0, split)} · ${goalTargetLabel(text.slice(split + 1), language)}`;
}
export function replayPatternFromFilename(filename) {
  const name = String(filename || '').split(/[\\/]/u).pop() || '';
  return name.match(/^([A-Za-z0-9]+(?:_[A-Za-z][A-Za-z0-9]*)*_(?:sum-)?\d+)(?=[_.]|$)/u)?.[1] || '';
}
// Only call for an accepted move. Remove its spawn to apply the post-move boundary.
export function sumGoalCompleted(target, state, selectedRate, dtype) {
  const goal = parseGoalTarget(target);
  if (goal?.kind !== 'sum' || restoreSuccessRate(selectedRate, dtype) !== 1) return false;
  if (state?.transition?.kind !== 'move') return false;
  const spawn = state.transition.metadata?.appear_tile;
  const total = state.board.reduce((sum, tile, index) => sum + (spawn?.index === index ? 0 : Number(tile)), 0);
  return total % 16384 >= goal.value - 2;
}
