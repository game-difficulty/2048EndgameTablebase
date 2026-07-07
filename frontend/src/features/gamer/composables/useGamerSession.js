import { onMounted, onUnmounted, ref, watch } from 'vue';

import { createLocalStorageStore } from '../../../services/storage/localStorageStore';
import { getEvilCore } from '../../../services/wasm/aiCoreClient';

const VALID_DIRECTIONS = new Set(['left', 'right', 'up', 'down']);
const DIRECTION_BY_CODE = {
  1: 'left',
  2: 'right',
  3: 'up',
  4: 'down',
};
const SPAWN_RATE4 = 0.1;
const GAMER_TOP_TILE = 32768;
const MAX_HISTORY_LENGTH = 1000;
const AI_WORKER_VERSION = 'worker-614a4af-20260706';

const gamerStore = createLocalStorageStore({
  key: 'gamer',
  version: 1,
  defaultValue: null,
});

function exponentToValue(exponent) {
  const exp = Number(exponent) || 0;
  return exp > 0 ? 2 ** exp : 0;
}

function valueToExponent(value) {
  const number = Number(value) || 0;
  if (number <= 0) return 0;
  if (number >= GAMER_TOP_TILE) return 15;
  const exponent = Math.log2(number);
  return Number.isInteger(exponent) && exponent > 0 ? Math.min(15, exponent) : 0;
}

function normalizeHex(hex) {
  return String(hex || '')
    .trim()
    .replace(/^0x/i, '')
    .replace(/[^0-9a-f]/gi, '')
    .padStart(16, '0')
    .slice(-16)
    .toLowerCase();
}

function boardFromHex(hex) {
  return normalizeHex(hex)
    .split('')
    .map((digit) => exponentToValue(Number.parseInt(digit, 16)));
}

function boardToHex(values) {
  return values
    .slice(0, 16)
    .map((value) => valueToExponent(value).toString(16))
    .join('');
}

function boardToEncoded(values) {
  let encoded = 0n;
  values.slice(0, 16).forEach((value, index) => {
    encoded |= BigInt(valueToExponent(value) & 0xf) << BigInt((15 - index) * 4);
  });
  return encoded;
}

function extractSpecialTiles(values) {
  return values.reduce((acc, value, index) => {
    const number = Number(value) || 0;
    if (number > GAMER_TOP_TILE) {
      acc.push({ index, value: number });
    }
    return acc;
  }, []);
}

function applySpecialTiles(values, tiles) {
  const next = values.slice(0, 16);
  for (const tile of Array.isArray(tiles) ? tiles : []) {
    const index = Number(tile?.index);
    const value = Number(tile?.value);
    if (Number.isInteger(index) && index >= 0 && index < 16 && value > GAMER_TOP_TILE) {
      next[index] = value;
    }
  }
  return next;
}

function linePositions(direction) {
  if (direction === 'left') {
    return Array.from({ length: 4 }, (_unused, row) =>
      Array.from({ length: 4 }, (__unused, col) => row * 4 + col)
    );
  }
  if (direction === 'right') {
    return Array.from({ length: 4 }, (_unused, row) =>
      Array.from({ length: 4 }, (__unused, offset) => row * 4 + (3 - offset))
    );
  }
  if (direction === 'up') {
    return Array.from({ length: 4 }, (_unused, col) =>
      Array.from({ length: 4 }, (__unused, row) => row * 4 + col)
    );
  }
  if (direction === 'down') {
    return Array.from({ length: 4 }, (_unused, col) =>
      Array.from({ length: 4 }, (__unused, offset) => (3 - offset) * 4 + col)
    );
  }
  return [];
}

function simulateLine(values) {
  const result = [0, 0, 0, 0];
  const distances = [0, 0, 0, 0];
  const pops = [0, 0, 0, 0];
  let scoreDelta = 0;

  const nonZero = values
    .map((value, index) => [index, Number(value) || 0])
    .filter(([_index, value]) => value !== 0);

  let read = 0;
  let write = 0;
  while (read < nonZero.length) {
    const [sourceIndex, value] = nonZero[read];
    if (read + 1 < nonZero.length && nonZero[read + 1][1] === value) {
      const [nextSourceIndex] = nonZero[read + 1];
      const mergedValue = value * 2;
      result[write] = mergedValue;
      scoreDelta += mergedValue;
      distances[sourceIndex] = sourceIndex - write;
      distances[nextSourceIndex] = nextSourceIndex - write;
      pops[write] = 1;
      read += 2;
    } else {
      result[write] = value;
      distances[sourceIndex] = sourceIndex - write;
      read += 1;
    }
    write += 1;
  }

  return { result, distances, pops, scoreDelta };
}

function simulateMove(values, direction) {
  const nextBoard = new Array(16).fill(0);
  const slideDistances = new Array(16).fill(0);
  const popPositions = new Array(16).fill(0);
  let scoreDelta = 0;

  for (const positions of linePositions(direction)) {
    const lineValues = positions.map((index) => Number(values[index]) || 0);
    const simulated = simulateLine(lineValues);
    scoreDelta += simulated.scoreDelta;

    positions.forEach((boardIndex, offset) => {
      slideDistances[boardIndex] = simulated.distances[offset];
      if (simulated.pops[offset]) {
        popPositions[boardIndex] = 1;
      }
      nextBoard[boardIndex] = simulated.result[offset];
    });
  }

  return {
    board: nextBoard,
    slideDistances,
    popPositions,
    scoreDelta,
  };
}

function boardsEqual(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function legalMoves(values) {
  return ['left', 'right', 'up', 'down'].filter((direction) => (
    !boardsEqual(simulateMove(values, direction).board, values)
  ));
}

function randomSpawn(values) {
  const emptyIndices = values
    .map((value, index) => (Number(value) === 0 ? index : null))
    .filter((index) => index !== null);
  if (!emptyIndices.length) {
    return null;
  }
  const index = emptyIndices[Math.floor(Math.random() * emptyIndices.length)];
  const exponent = Math.random() < SPAWN_RATE4 ? 2 : 1;
  return {
    index,
    value: exponentToValue(exponent),
  };
}

function aiSpeedRatio(speed) {
  const normalized = Number(speed);
  const clamped = Number.isFinite(normalized) ? Math.max(0, Math.min(200, normalized)) : 100;
  return 10 ** ((100 - clamped) / 100);
}

function historyEntry(values, currentScore, specialTiles) {
  return {
    board: values.slice(0, 16),
    score: Number(currentScore) || 0,
    specialTiles: Array.isArray(specialTiles) ? specialTiles.map((tile) => ({ ...tile })) : [],
  };
}

export function useGamerSession(activeRef) {
  const board = ref(new Array(16).fill(0));
  const metadata = ref(null);
  const score = ref({ current: 0, best: 0 });
  const wsStatus = ref('connected');
  const aiEnabled = ref(false);
  const difficulty = ref(0);
  const aiSpeed = ref(100);
  const hexInput = ref('');
  const currentHex = ref('0000000000000000');
  const specialTiles = ref([]);
  const scoreAnimations = ref([]);
  const aiWorkerReady = ref(false);

  let animIdCounter = 0;
  let aiContinuationTimer = null;
  let aiRunning = false;
  let evilCoreModule = null;
  let aiWorker = null;
  let pendingAiMove = null;
  let evilGen = null;
  let lastAiSpeedRatio = null;
  let persistTimer = null;
  const history = [];

  const writePersistedState = () => {
    gamerStore.write({
      board: board.value.slice(0, 16),
      metadata: null,
      score: score.value,
      difficulty: difficulty.value,
      aiSpeed: aiSpeed.value,
      currentHex: currentHex.value,
      specialTiles: specialTiles.value,
    });
  };

  const clearPersistTimer = () => {
    if (persistTimer !== null) {
      window.clearTimeout(persistTimer);
      persistTimer = null;
    }
  };

  const persistState = ({ immediate = false } = {}) => {
    if (immediate || !aiEnabled.value) {
      clearPersistTimer();
      writePersistedState();
      return;
    }
    if (persistTimer !== null) {
      return;
    }
    persistTimer = window.setTimeout(() => {
      persistTimer = null;
      writePersistedState();
    }, 1000);
  };

  const flushPersistState = () => {
    clearPersistTimer();
    writePersistedState();
  };

  const pushHistory = () => {
    history.push(historyEntry(board.value, score.value.current, specialTiles.value));
    if (history.length > MAX_HISTORY_LENGTH) {
      history.splice(0, history.length - MAX_HISTORY_LENGTH);
    }
  };

  const syncDerivedState = () => {
    currentHex.value = boardToHex(board.value);
    specialTiles.value = extractSpecialTiles(board.value);
  };

  const applyBoardState = (values, animation = null, nextScore = score.value.current) => {
    board.value = applySpecialTiles(values, specialTiles.value);
    metadata.value = animation;
    score.value = {
      current: Number(nextScore) || 0,
      best: Math.max(Number(score.value.best) || 0, Number(nextScore) || 0),
    };
    syncDerivedState();
    persistState();
  };

  const clearAiContinuationTimer = () => {
    if (aiContinuationTimer !== null) {
      window.clearTimeout(aiContinuationTimer);
      aiContinuationTimer = null;
    }
  };

  const stopAI = () => {
    aiEnabled.value = false;
    aiRunning = false;
    clearAiContinuationTimer();
    flushPersistState();
  };

  const canContinueAI = () => aiEnabled.value && activeRef?.value && wsStatus.value === 'connected';

  const scheduleAiStep = (delay = 0) => {
    clearAiContinuationTimer();
    if (!canContinueAI() || aiRunning) {
      return;
    }
    aiContinuationTimer = window.setTimeout(async () => {
      aiContinuationTimer = null;
      if (!canContinueAI() || aiRunning) {
        return;
      }
      await runAiStep(true);
    }, delay);
  };

  const spawnEvil = async (values) => {
    const fallback = () => randomSpawn(values);
    try {
      const module = evilCoreModule || await getEvilCore();
      evilCoreModule = module;
      if (!module?.EvilGen) {
        return fallback();
      }
      const encoded = boardToEncoded(values);
      evilGen = evilGen || new module.EvilGen(encoded);
      evilGen.reset_board(encoded);
      const result = evilGen.gen_new_num(5);
      const index = Number(result?.[1]);
      const exponent = Number(result?.[2]);
      if (!Number.isInteger(index) || index < 0 || index >= 16 || exponent <= 0 || values[index] !== 0) {
        return fallback();
      }
      return {
        index,
        value: exponentToValue(exponent),
      };
    } catch (error) {
      console.error('EvilGen WASM spawn failed; falling back to random spawn.', error);
      return fallback();
    }
  };

  const syncAiSpeedToWorker = (worker = aiWorker, force = false) => {
    if (!worker) {
      return;
    }
    const ratio = aiSpeedRatio(aiSpeed.value);
    if (!force && lastAiSpeedRatio !== null && Math.abs(ratio - lastAiSpeedRatio) < 0.000001) {
      return;
    }
    lastAiSpeedRatio = ratio;
    worker.postMessage({
      type: 'update_speed',
      ratio,
    });
  };

  const disposeAiWorker = () => {
    if (pendingAiMove) {
      pendingAiMove.reject(new Error('AI worker disposed.'));
      pendingAiMove = null;
    }
    aiWorker?.terminate();
    aiWorker = null;
    aiWorkerReady.value = false;
    lastAiSpeedRatio = null;
  };

  const ensureAiWorker = () => {
    if (aiWorker) {
      return aiWorker;
    }

    aiWorker = new Worker(`/wasm/ai_worker.js?v=${AI_WORKER_VERSION}`, { type: 'module' });
    syncAiSpeedToWorker(aiWorker, true);
    aiWorker.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === 'ready') {
        aiWorkerReady.value = true;
        return;
      }
      if (data.type === 'move_result' && pendingAiMove) {
        const pending = pendingAiMove;
        pendingAiMove = null;
        window.clearTimeout(pending.timeoutId);
        pending.resolve(Number(data.best_move) || 0);
      }
    };
    aiWorker.onerror = (event) => {
      aiWorkerReady.value = false;
      if (pendingAiMove) {
        const pending = pendingAiMove;
        pendingAiMove = null;
        window.clearTimeout(pending.timeoutId);
        pending.reject(new Error(event.message || 'AI worker error.'));
      }
    };
    return aiWorker;
  };

  const requestWorkerAiMove = () => new Promise((resolve, reject) => {
    const worker = ensureAiWorker();
    if (pendingAiMove) {
      pendingAiMove.reject(new Error('AI worker request superseded.'));
      window.clearTimeout(pendingAiMove.timeoutId);
      pendingAiMove = null;
    }
    const timeoutId = window.setTimeout(() => {
      if (!pendingAiMove) return;
      pendingAiMove = null;
      reject(new Error('AI worker timed out.'));
    }, 60000);
    pendingAiMove = { resolve, reject, timeoutId };
    syncAiSpeedToWorker(worker);
    worker.postMessage({
      type: 'calculate',
      board_encoded: boardToHex(board.value),
    });
  });

  const chooseAiMove = async () => {
    const moves = legalMoves(board.value);
    if (!moves.length) {
      return null;
    }

    try {
      const candidate = DIRECTION_BY_CODE[await requestWorkerAiMove()];
      return moves.includes(candidate) ? candidate : moves[0];
    } catch (error) {
      console.error('AI worker step failed; using first legal move.', error);
      return moves[0];
    }
  };

  const moveBoard = async (direction, source = 'manual') => {
    if (!VALID_DIRECTIONS.has(direction)) {
      return false;
    }

    const before = board.value.slice(0, 16);
    const simulated = simulateMove(before, direction);
    if (boardsEqual(simulated.board, before)) {
      if (source === 'ai') {
        stopAI();
      }
      return false;
    }

    const nextScore = score.value.current + simulated.scoreDelta;
    const spawn = Math.random() > (Number(difficulty.value) || 0) / 100
      ? randomSpawn(simulated.board)
      : await spawnEvil(simulated.board);
    const nextBoard = simulated.board.slice(0, 16);
    if (spawn) {
      nextBoard[spawn.index] = spawn.value;
    }

    specialTiles.value = extractSpecialTiles(nextBoard);
    applyBoardState(nextBoard, {
      direction,
      slide_distances: simulated.slideDistances,
      pop_positions: simulated.popPositions,
      appear_tile: spawn
        ? {
            index: spawn.index,
            value: spawn.value,
          }
        : null,
    }, nextScore);
    pushHistory();
    return true;
  };

  const newGame = () => {
    stopAI();
    const nextBoard = new Array(16).fill(0);
    const first = randomSpawn(nextBoard);
    if (first) nextBoard[first.index] = first.value;
    const second = randomSpawn(nextBoard);
    if (second) nextBoard[second.index] = second.value;

    specialTiles.value = [];
    board.value = nextBoard;
    metadata.value = null;
    score.value = {
      current: 0,
      best: Number(score.value.best) || 0,
    };
    syncDerivedState();
    history.length = 0;
    pushHistory();
    persistState();
  };

  const undo = () => {
    if (history.length <= 1) {
      return false;
    }
    stopAI();
    history.pop();
    const previous = history[history.length - 1];
    specialTiles.value = previous.specialTiles;
    board.value = applySpecialTiles(previous.board, previous.specialTiles);
    metadata.value = null;
    score.value = {
      current: Number(previous.score) || 0,
      best: Number(score.value.best) || 0,
    };
    syncDerivedState();
    persistState();
    return true;
  };

  const setBoardFromHex = (hex) => {
    const normalized = normalizeHex(hex);
    if (!normalized) {
      return false;
    }
    stopAI();
    specialTiles.value = [];
    board.value = boardFromHex(normalized);
    metadata.value = null;
    score.value = {
      current: 0,
      best: Number(score.value.best) || 0,
    };
    syncDerivedState();
    history.length = 0;
    pushHistory();
    persistState();
    return true;
  };

  const runAiStep = async (fromContinuousAI = false) => {
    if (aiRunning) {
      return false;
    }
    aiRunning = true;
    let shouldContinue = false;
    try {
      const direction = await chooseAiMove();
      if (!direction) {
        stopAI();
        return false;
      }
      const moved = await moveBoard(direction, fromContinuousAI ? 'ai' : 'manual-ai-step');
      shouldContinue = Boolean(moved && fromContinuousAI && canContinueAI());
      return moved;
    } finally {
      aiRunning = false;
      if (shouldContinue) {
        scheduleAiStep(0);
      }
    }
  };

  const triggerAction = (action, payload = {}) => {
    if (action === 'INIT_GAME') {
      newGame();
      return true;
    }
    if (action === 'UNDO') {
      return undo();
    }
    if (action === 'SET_BOARD') {
      return setBoardFromHex(payload.hex_str);
    }
    if (action === 'USER_MOVE') {
      moveBoard(String(payload.dir || '').toLowerCase(), payload.source || 'manual');
      return true;
    }
    if (action === 'AI_STEP') {
      runAiStep(false);
      return true;
    }
    if (action === 'SAVE_GAME_STATE') {
      persistState();
      return true;
    }
    return false;
  };

  const toggleAI = () => {
    if (aiEnabled.value) {
      stopAI();
      return;
    }
    aiEnabled.value = true;
    scheduleAiStep();
  };

  const updateSettings = () => {
    difficulty.value = Math.max(0, Math.min(100, Number(difficulty.value) || 0));
    aiSpeed.value = Math.max(0, Math.min(200, Number(aiSpeed.value) || 0));
    syncAiSpeedToWorker();
    persistState({ immediate: true });
  };

  const setBoard = () => {
    const value = hexInput.value.trim();
    if (!value) return;
    setBoardFromHex(value);
  };

  const writeCurrentBoardToHex = () => {
    hexInput.value = boardToHex(board.value);
  };

  const openBrowserAi = () => false;

  const loadSavedState = () => {
    const saved = gamerStore.read();
    if (!saved?.board || !Array.isArray(saved.board) || saved.board.length !== 16) {
      newGame();
      return;
    }
    difficulty.value = Math.max(0, Math.min(100, Number(saved.difficulty) || 0));
    aiSpeed.value = Math.max(0, Math.min(200, Number(saved.aiSpeed) || 100));
    specialTiles.value = Array.isArray(saved.specialTiles) ? saved.specialTiles : [];
    board.value = applySpecialTiles(saved.board, specialTiles.value);
    metadata.value = null;
    score.value = {
      current: Number(saved.score?.current) || 0,
      best: Number(saved.score?.best) || 0,
    };
    syncDerivedState();
    history.length = 0;
    pushHistory();
  };

  const prewarmWasmEngines = () => {
    ensureAiWorker();
    getEvilCore()
      .then((module) => {
        evilCoreModule = module;
        if (module?.EvilGen) {
          const warmupEvilGen = new module.EvilGen(0n);
          warmupEvilGen.gen_new_num(1);
          warmupEvilGen.delete?.();
        }
      })
      .catch((error) => {
        console.error('Failed to prewarm EvilGen WASM module.', error);
      });
  };

  const handleKeydown = (event) => {
    if (!activeRef?.value) return;
    const target = event.target;
    if (
      target instanceof HTMLElement &&
      (target.tagName === 'INPUT' ||
        target.tagName === 'SELECT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable)
    ) {
      return;
    }

    const map = {
      ArrowUp: 'up',
      KeyW: 'up',
      ArrowDown: 'down',
      KeyS: 'down',
      ArrowLeft: 'left',
      KeyA: 'left',
      ArrowRight: 'right',
      KeyD: 'right',
    };

    if (map[event.code]) {
      event.preventDefault();
      triggerAction('USER_MOVE', { dir: map[event.code] });
    } else if (event.code === 'Enter' || event.code === 'Space') {
      event.preventDefault();
      if (aiEnabled.value) {
        stopAI();
      } else {
        triggerAction('AI_STEP');
      }
    } else if (event.code === 'Backspace' || event.code === 'Delete') {
      event.preventDefault();
      if (aiEnabled.value) {
        stopAI();
      } else {
        triggerAction('UNDO');
      }
    }
  };

  const handleBeforeUnload = () => {
    flushPersistState();
  };

  watch(() => score.value.current, (newVal, oldVal) => {
    if (newVal > oldVal && oldVal !== undefined && oldVal !== 0) {
      const diff = newVal - oldVal;
      const id = animIdCounter++;
      scoreAnimations.value.push({ id, value: diff });
      window.setTimeout(() => {
        scoreAnimations.value = scoreAnimations.value.filter((item) => item.id !== id);
      }, 1000);
    }
  });

  watch(
    activeRef,
    (isActive) => {
      if (isActive && aiEnabled.value) {
        scheduleAiStep();
      } else {
        clearAiContinuationTimer();
      }
    },
    { immediate: true }
  );

  onMounted(() => {
    loadSavedState();
    prewarmWasmEngines();
    window.addEventListener('keydown', handleKeydown);
    window.addEventListener('beforeunload', handleBeforeUnload);
  });

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown);
    window.removeEventListener('beforeunload', handleBeforeUnload);
    stopAI();
    disposeAiWorker();
    if (evilGen?.delete) {
      evilGen.delete();
      evilGen = null;
    }
    persistState({ immediate: true });
  });

  return {
    board,
    metadata,
    score,
    wsStatus,
    aiEnabled,
    difficulty,
    aiSpeed,
    hexInput,
    scoreAnimations,
    aiWorkerReady,
    triggerAction,
    toggleAI,
    updateSettings,
    setBoard,
    writeCurrentBoardToHex,
    openBrowserAi,
  };
}
