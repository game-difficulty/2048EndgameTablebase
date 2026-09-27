import { NODE_TILES, VARIANTS } from './engine.js';

const KEY = 'human-timer-splits-v1';
export const DEFAULT_TIMER_SPLITS = Object.freeze(Object.fromEntries(Object.keys(VARIANTS).map(variant => [variant, NODE_TILES.map(String)])));

export function parseTimerSplit(value) {
  const text = String(value || '').trim();
  const parts = text.split('+');
  if (!text || parts.length > 8 || parts.some(part => !/^\d+$/.test(part))) throw new Error('invalid_timer_splits');
  const values = parts.map(Number);
  if (values.some(number => number < 2 || number > 2 ** 31 || !Number.isInteger(Math.log2(number))) || values.some((number, index) => index && values[index - 1] < number)) throw new Error('invalid_timer_splits');
  return { key: values.join('+'), values };
}

export function normalizeTimerSplits(value) {
  const result = {};
  for (const variant of Object.keys(VARIANTS)) {
    const input = value?.[variant] ?? DEFAULT_TIMER_SPLITS[variant];
    if (!Array.isArray(input) || input.length > 32) throw new Error('invalid_timer_splits');
    result[variant] = input.map(item => parseTimerSplit(item).key);
    if (new Set(result[variant]).size !== result[variant].length) throw new Error('invalid_timer_splits');
  }
  return result;
}

export function readTimerSplits() {
  try { return normalizeTimerSplits(JSON.parse(localStorage.getItem(KEY) || 'null')); }
  catch { return normalizeTimerSplits(DEFAULT_TIMER_SPLITS); }
}
export function saveTimerSplits(value) {
  const normalized = normalizeTimerSplits(value); localStorage.setItem(KEY, JSON.stringify(normalized)); return normalized;
}
export function timerSplitsFor(variant) { return readTimerSplits()[variant] || DEFAULT_TIMER_SPLITS[variant]; }

export function timerSplitReached(board, expression) {
  const { values } = parseTimerSplit(expression);
  const counts = new Map(); board.forEach(value => counts.set(value, (counts.get(value) || 0) + 1));
  for (const requested of values) {
    if (counts.get(requested)) counts.set(requested, counts.get(requested) - 1);
    else if ([...counts].some(([value, count]) => count > 0 && value > requested)) return true;
    else return false;
  }
  return true;
}

export function timerSplitRow(expression) {
  const parsed = parseTimerSplit(expression);
  return { key: parsed.key, tile: parsed.values.at(-1), depth: parsed.values.length - 1 };
}
