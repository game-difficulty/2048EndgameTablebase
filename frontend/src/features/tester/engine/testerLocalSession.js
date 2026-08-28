import { restoreSuccessRate } from '../../../utils/successRate.js';
import { encodeBoard } from '../../replay/engine/replayTransition.js';
import { PERFORMANCE_LABELS } from '../../replay/engine/replayAnalysis.js';
import {
  createPracticeSession,
  reducePracticeSession,
} from '../../practice/engine/practiceSession.js';
import { buildOptimisticTesterLastStep } from './testerOptimisticFeedback.js';

const DIRECTIONS = Object.freeze(['left', 'right', 'up', 'down']);
const DIRECTION_BITS = Object.freeze({ left: 0, right: 1, up: 2, down: 3 });
const MAX_RECORD_MOVES = 3999;
const SENTINEL = Object.freeze({
  change: 88,
  rates: [666666666, 233333333, 314159265, 987654321],
});

const emptyCounts = (labels = PERFORMANCE_LABELS) => Object.fromEntries(
  labels.map((label) => [label, 0]),
);

const emptyLastStep = () => ({
  board: [],
  board_lines: [],
  result_lines: [],
  results: {},
  dtype: '?',
  message_lines: [],
  evaluation: '',
  direction: null,
  best_move: null,
  loss: null,
  goodness_of_fit: null,
});

const compactRate = (value, dtype) => {
  const restored = restoreSuccessRate(value, dtype);
  if (restored == null || !Number.isFinite(restored)) return 0;
  return Math.max(0, Math.min(0xffffffff, Math.trunc(restored * 4e9))) >>> 0;
};

const recordForMove = ({ board, direction, spawnIndex, spawnValue, results, dtype }) => ({
  board: encodeBoard(board),
  change: (
    ((DIRECTION_BITS[direction] & 0b11) << 5)
    | ((spawnIndex & 0b1111) << 1)
    | (spawnValue === 4 ? 1 : 0)
  ),
  rates: DIRECTIONS.map((item) => compactRate(results?.[item], dtype)),
});

const writeUint64LE = (view, offset, value) => {
  const normalized = BigInt.asUintN(64, BigInt(value || 0n));
  view.setUint32(offset, Number(normalized & 0xffffffffn), true);
  view.setUint32(offset + 4, Number((normalized >> 32n) & 0xffffffffn), true);
};

const writeRecord = (view, offset, record) => {
  writeUint64LE(view, offset, record.board);
  view.setUint8(offset + 8, Number(record.change) & 0xff);
  record.rates.forEach((value, index) => {
    view.setUint32(offset + 9 + index * 4, Number(value) >>> 0, true);
  });
};

export function createTesterLocalSession({
  board,
  boardHex,
  useVariant = false,
  context = null,
  performanceLabels = PERFORMANCE_LABELS,
  openingLogs = [],
} = {}) {
  return {
    practice: createPracticeSession({ board, boardHex, useVariant, context }),
    metrics: {
      combo: 0,
      max_combo: 0,
      goodness_of_fit: 1,
      performance_stats: emptyCounts(performanceLabels),
      score: 0,
      best_score: 0,
    },
    lastStep: emptyLastStep(),
    logs: Array.isArray(openingLogs) ? [...openingLogs] : [],
    records: [],
  };
}

export function applyTesterLocalMove(session, {
  direction,
  results,
  dtype,
  spawnRate4,
  randomSource,
  nextContext,
} = {}) {
  if (!session?.practice) return { session, accepted: false, reason: 'session_required' };
  const normalizedDirection = String(direction || '').toLowerCase();
  const lastStep = buildOptimisticTesterLastStep({
    board: session.practice.board,
    results,
    dtype,
    direction: normalizedDirection,
    goodnessOfFit: session.metrics.goodness_of_fit,
  });
  if (!lastStep) return { session, accepted: false, reason: 'results_required' };
  const reduced = reducePracticeSession(session.practice, {
    type: 'MOVE_RANDOM',
    direction: normalizedDirection,
    spawnRate4,
    randomSource,
    nextContext,
  });
  if (!reduced.accepted) return { session, accepted: false, reason: reduced.reason };

  const perfect = Number(lastStep.loss) <= 3e-10;
  const combo = perfect ? session.metrics.combo + 1 : 0;
  const performanceStats = { ...session.metrics.performance_stats };
  performanceStats[lastStep.evaluation] = Number(performanceStats[lastStep.evaluation] || 0) + 1;
  const record = recordForMove({
    board: session.practice.board,
    direction: normalizedDirection,
    spawnIndex: reduced.state.transition.metadata?.appear_tile?.index,
    spawnValue: reduced.state.transition.metadata?.appear_tile?.value,
    results,
    dtype,
  });
  const records = session.records.length < MAX_RECORD_MOVES
    ? [...session.records, record]
    : session.records;
  const message = perfect
    ? `${lastStep.evaluation} Combo: ${combo}x`
    : `${lastStep.evaluation} | loss ${Number(lastStep.loss).toFixed(4)} | GOF ${Number(lastStep.goodness_of_fit).toFixed(4)}`;
  const next = {
    ...session,
    practice: reduced.state,
    lastStep: {
      ...lastStep,
      message_lines: [message],
    },
    metrics: {
      combo,
      max_combo: Math.max(session.metrics.max_combo, combo),
      goodness_of_fit: lastStep.goodness_of_fit,
      performance_stats: performanceStats,
      score: reduced.state.score,
      best_score: Math.max(session.metrics.best_score, reduced.state.score),
    },
    logs: [...session.logs, message],
    records,
  };
  return { session: next, accepted: true, reason: null };
}

export function replaceTesterLocalBoard(session, { board, boardHex, context = null } = {}) {
  if (!session?.practice) return session;
  const reduced = reducePracticeSession(session.practice, {
    type: 'SET_BOARD',
    board,
    boardHex,
    nextContext: context,
  });
  return createTesterLocalSession({
    board: reduced.state.board,
    useVariant: reduced.state.useVariant,
    context: reduced.state.context,
    performanceLabels: Object.keys(session.metrics.performance_stats || {}),
    openingLogs: session.logs.slice(0, 1),
  });
}

export function encodeTesterReplay(session) {
  const records = Array.isArray(session?.records) ? session.records : [];
  if (!records.length) return null;
  const bytes = new ArrayBuffer((records.length + 1) * 25);
  const view = new DataView(bytes);
  records.forEach((record, index) => writeRecord(view, index * 25, record));
  writeRecord(view, records.length * 25, {
    board: encodeBoard(session.practice.board),
    change: SENTINEL.change,
    rates: SENTINEL.rates,
  });
  return bytes;
}

export function testerReplayFilename(fullPattern, goodnessOfFit) {
  const pattern = String(fullPattern || 'tester')
    .replace(/[^A-Za-z0-9_-]+/gu, '_')
    .replace(/^[-_]+|[-_]+$/gu, '') || 'tester';
  const goodness = Math.max(0, Math.min(1, Number(goodnessOfFit) || 0));
  return `${pattern}_${goodness.toFixed(4)}.rpl`;
}
