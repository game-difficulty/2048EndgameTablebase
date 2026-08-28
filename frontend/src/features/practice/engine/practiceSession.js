import {
  boardHex,
  buildOptimisticMoveOnlyTransition,
  buildOptimisticMoveTransition,
  decodeBoard,
  encodeBoard,
} from '../../replay/engine/replayTransition.js';

const VALID_DIRECTIONS = new Set(['up', 'down', 'left', 'right']);

const cloneBoard = (board) => (
  Array.isArray(board)
    ? board.slice(0, 16).map((value) => Number(value) || 0)
    : new Array(16).fill(0)
);

const cloneContext = (context) => {
  if (context == null || typeof context !== 'object') return context ?? null;
  if (typeof structuredClone === 'function') return structuredClone(context);
  return JSON.parse(JSON.stringify(context));
};

const normalizedHex = (value) => {
  const text = String(value || '').trim().replace(/^0x/iu, '').toLowerCase();
  return /^[0-9a-f]{1,16}$/u.test(text) ? text.padStart(16, '0') : null;
};

const boardFromInput = ({ board, boardHex: encoded }) => {
  if (Array.isArray(board) && board.length >= 16) return cloneBoard(board);
  const normalized = normalizedHex(encoded);
  return normalized ? decodeBoard(BigInt(`0x${normalized}`)) : new Array(16).fill(0);
};

const scoreDeltaForTransition = (transition) => {
  const pops = transition?.metadata?.pop_positions || [];
  return pops.reduce((total, marker, index) => (
    marker ? total + (Number(transition.board?.[index]) || 0) : total
  ), 0);
};

const makeHistoryEntry = ({ board, score, context, lastMove = null }) => ({
  board: cloneBoard(board),
  boardHex: boardHex(encodeBoard(board)),
  score: Math.max(0, Math.trunc(Number(score) || 0)),
  context: cloneContext(context),
  lastMove: VALID_DIRECTIONS.has(lastMove) ? lastMove : null,
});

const makeTransition = (revision, kind, fromBoard, toBoard, metadata = null) => ({
  id: revision,
  revision,
  kind,
  fromBoard: cloneBoard(fromBoard),
  toBoard: cloneBoard(toBoard),
  metadata: metadata && typeof metadata === 'object' ? { ...metadata } : null,
});

const nextState = (state, {
  board,
  score = state.score,
  context = state.context,
  phase = 'ready',
  history = state.history,
  pendingTurn = null,
  transitionKind = 'snapshot',
  metadata = null,
}) => {
  const revision = state.revision + 1;
  const normalizedBoard = cloneBoard(board);
  return {
    ...state,
    revision,
    board: normalizedBoard,
    boardHex: boardHex(encodeBoard(normalizedBoard)),
    score: Math.max(0, Math.trunc(Number(score) || 0)),
    context: cloneContext(context),
    phase,
    history,
    pendingTurn,
    transition: makeTransition(
      revision,
      transitionKind,
      state.board,
      normalizedBoard,
      metadata,
    ),
  };
};

const reject = (state, reason) => ({ state, accepted: false, reason });
const accept = (state) => ({ state, accepted: true, reason: null });

const resolveNextContext = (nextContext, fallback) => (
  typeof nextContext === 'function'
    ? nextContext()
    : (nextContext ?? fallback)
);

export function selectRandomSpawn(board, spawnRate4 = 0.1, randomSource = Math.random) {
  const empty = cloneBoard(board)
    .map((value, index) => (value === 0 ? index : -1))
    .filter((index) => index >= 0);
  if (!empty.length) return null;
  const positionRoll = Math.max(0, Math.min(0.999999999999, Number(randomSource()) || 0));
  const index = empty[Math.min(empty.length - 1, Math.floor(positionRoll * empty.length))];
  const value = Number(randomSource()) < Math.max(0, Math.min(1, Number(spawnRate4) || 0))
    ? 4
    : 2;
  return { index, value };
}

export function createPracticeSession({
  board,
  boardHex: encoded,
  score = 0,
  context = null,
  useVariant = false,
  revision = 0,
} = {}) {
  const initialBoard = boardFromInput({ board, boardHex: encoded });
  const initialScore = Math.max(0, Math.trunc(Number(score) || 0));
  const initialContext = cloneContext(context);
  return {
    revision: Math.max(0, Math.trunc(Number(revision) || 0)),
    board: initialBoard,
    boardHex: boardHex(encodeBoard(initialBoard)),
    score: initialScore,
    context: initialContext,
    useVariant: Boolean(useVariant),
    phase: 'ready',
    history: [makeHistoryEntry({
      board: initialBoard,
      score: initialScore,
      context: initialContext,
    })],
    pendingTurn: null,
    transition: null,
  };
}

export function canApplyPracticeSeed(state, requestRevision, responseRevision) {
  if (!state || !Number.isInteger(Number(state.revision))) return false;
  if (!Number.isInteger(requestRevision) || !Number.isInteger(responseRevision)) return false;
  const requested = requestRevision;
  const echoed = responseRevision;
  return (
    Number.isInteger(requested)
    && requested >= 0
    && echoed === requested
    && Number(state.revision) === requested
  );
}

export function reducePracticeSession(state, command = {}) {
  if (!state || !Array.isArray(state.board)) throw new TypeError('practice_state_required');
  const expectedRevision = command.expectedRevision;
  if (
    expectedRevision != null
    && Number(expectedRevision) !== Number(state.revision)
  ) {
    return reject(state, 'stale_revision');
  }

  switch (String(command.type || '').toUpperCase()) {
    case 'MOVE_RANDOM': {
      if (state.phase !== 'ready') return reject(state, 'session_not_ready');
      const direction = String(command.direction || '').toLowerCase();
      if (!VALID_DIRECTIONS.has(direction)) return reject(state, 'invalid_direction');
      const transition = buildOptimisticMoveTransition(
        state.board,
        direction,
        state.useVariant,
        command.spawnRate4,
        command.randomSource,
      );
      if (!transition) return reject(state, 'invalid_move');
      const score = state.score + scoreDeltaForTransition(transition);
      const context = resolveNextContext(command.nextContext, state.context);
      const history = [...state.history, makeHistoryEntry({
        board: transition.board,
        score,
        context,
        lastMove: direction,
      })];
      return accept(nextState(state, {
        board: transition.board,
        score,
        context,
        history,
        transitionKind: 'move',
        metadata: transition.metadata,
      }));
    }

    case 'MOVE_ONLY': {
      if (state.phase !== 'ready') return reject(state, 'session_not_ready');
      const direction = String(command.direction || '').toLowerCase();
      if (!VALID_DIRECTIONS.has(direction)) return reject(state, 'invalid_direction');
      const transition = buildOptimisticMoveOnlyTransition(
        state.board,
        direction,
        state.useVariant,
      );
      if (!transition) return reject(state, 'invalid_move');
      const scoreDelta = scoreDeltaForTransition(transition);
      return accept(nextState(state, {
        board: transition.board,
        score: state.score + scoreDelta,
        phase: 'awaiting_spawn',
        pendingTurn: {
          baseRevision: state.revision,
          direction,
          scoreDelta,
        },
        transitionKind: 'move',
        metadata: transition.metadata,
      }));
    }

    case 'SPAWN': {
      if (state.phase !== 'awaiting_spawn' || !state.pendingTurn) {
        return reject(state, 'spawn_not_expected');
      }
      const index = Math.trunc(Number(command.index));
      const value = Math.trunc(Number(command.value));
      if (index < 0 || index >= 16 || ![2, 4].includes(value) || state.board[index] !== 0) {
        return reject(state, 'invalid_spawn');
      }
      const board = cloneBoard(state.board);
      board[index] = value;
      const context = resolveNextContext(command.nextContext, state.context);
      const history = [...state.history, makeHistoryEntry({
        board,
        score: state.score,
        context,
        lastMove: state.pendingTurn.direction,
      })];
      return accept(nextState(state, {
        board,
        context,
        history,
        transitionKind: 'spawn',
        metadata: { appear_tile: { index, value } },
      }));
    }

    case 'UNDO': {
      if (state.phase === 'awaiting_spawn') {
        const current = state.history.at(-1);
        return accept(nextState(state, {
          board: current.board,
          score: current.score,
          context: current.context,
          history: state.history,
          transitionKind: 'snapshot',
        }));
      }
      if (state.history.length <= 1) return reject(state, 'history_empty');
      const history = state.history.slice(0, -1);
      const previous = history.at(-1);
      return accept(nextState(state, {
        board: previous.board,
        score: previous.score,
        context: previous.context,
        history,
        transitionKind: 'snapshot',
      }));
    }

    case 'SET_BOARD': {
      const board = boardFromInput({ board: command.board, boardHex: command.boardHex });
      const score = command.score == null ? 0 : command.score;
      const context = resolveNextContext(command.nextContext, state.context);
      const history = [makeHistoryEntry({ board, score, context })];
      return accept(nextState(state, {
        board,
        score,
        context,
        history,
        transitionKind: 'snapshot',
      }));
    }

    default:
      return reject(state, 'unknown_command');
  }
}
