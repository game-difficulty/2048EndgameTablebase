import { computed, onMounted, onUnmounted, ref, watch } from 'vue';

import { KEYBOARD_OWNERS, keyboardInputAllowed } from '../../../app/keyboardOwnership';
import { useAppSettingsStore } from '../../../app/useAppSettings';
import { useAuthState } from '../../../services/auth/authState';
import { createLocalStorageStore } from '../../../services/storage/localStorageStore';
import { createSessionStorageStore } from '../../../services/storage/sessionStorageStore';
import { getEvilCore } from '../../../services/wasm/aiCoreClient';
import {
  createRankedInitialBoard,
  randomSpawnWithRng,
  Xoshiro128StarStar,
} from '../engine/seededRng';
import {
  DIRECTION_TO_NEXT_CODE,
  encodeRankedReplay,
  exactBoardCodes,
  MAX_RANKED_MOVES,
  RANKED_RECORD,
} from '../engine/rankedReplayEncoder';
import {
  abandonRankedRun,
  createRankedLeaseToken,
  createRankedRequestId,
  createRankedRun,
  fetchRankedRun,
  heartbeatRankedRun,
  submitRankedRun,
} from '../services/rankedRunClient';
import { createRankedRunLock } from '../services/rankedRunLock';

const VALID_DIRECTIONS = new Set(['left', 'right', 'up', 'down']);
const DIRECTION_BY_CODE = {
  1: 'left',
  2: 'right',
  3: 'up',
  4: 'down',
};
const SPAWN_RATE4 = 0.1;
const MIN_RANKED_SPAWN_RATE4 = 0.1;
const MAX_RANKED_SPAWN_RATE4 = 0.8;
const GAMER_TOP_TILE = 32768;
const MAX_HISTORY_LENGTH = 1000;
const AI_WORKER_VERSION = 'worker-614a4af-20260706';

const legacyGamerStore = createLocalStorageStore({
  key: 'gamer',
  version: 2,
  defaultValue: null,
  migrate: (value) => (value ? { ...value, ranked: null } : null),
});

const gamerPreferencesStore = createLocalStorageStore({
  key: 'gamer-preferences',
  version: 1,
  defaultValue: null,
});

const gamerSessionStore = createSessionStorageStore({
  key: 'gamer-session',
  version: 1,
  defaultValue: null,
});

const emptyRankedState = (overrides = {}) => ({
  runId: null,
  rulesVersion: 1,
  seedHex: '',
  rngState: null,
  initialTiles: [],
  eligible: false,
  usedUndo: false,
  usedSetBoard: false,
  usedAi: false,
  allDifficulty100: true,
  moveCount: 0,
  records: [],
  lastDifficulty: null,
  lastEventAt: Date.now(),
  status: 'unranked',
  errorCode: '',
  userId: null,
  spawnRate4: null,
  leaseToken: '',
  leaseExpiresAt: null,
  ...overrides,
});

const ulebLength = (rawValue) => {
  let value = Math.max(0, Math.floor(Number(rawValue) || 0));
  let length = 1;
  while (value >= 128) {
    value = Math.floor(value / 128);
    length += 1;
  }
  return length;
};

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

function randomSpawn(values, spawnRate4 = SPAWN_RATE4) {
  const emptyIndices = values
    .map((value, index) => (Number(value) === 0 ? index : null))
    .filter((index) => index !== null);
  if (!emptyIndices.length) {
    return null;
  }
  const index = emptyIndices[Math.floor(Math.random() * emptyIndices.length)];
  const exponent = Math.random() < spawnRate4 ? 2 : 1;
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
  const { config: appConfig } = useAppSettingsStore();
  const { ready: authReady, user: authUser } = useAuthState();
  const board = ref(new Array(16).fill(0));
  const metadata = ref(null);
  const transition = ref(null);
  const score = ref({ current: 0, best: 0 });
  const wsStatus = ref('connected');
  const aiEnabled = ref(false);
  const difficulty = ref(0);
  const aiSpeed = ref(100);
  const rankedParticipationEnabled = ref(true);
  const hexInput = ref('');
  const currentHex = ref('0000000000000000');
  const specialTiles = ref([]);
  const scoreAnimations = ref([]);
  const aiWorkerReady = ref(false);
  const ranked = ref(emptyRankedState());
  const rankedStatus = computed(() => ranked.value.status);
  const rankedMode = computed(() => {
    if (ranked.value.moveCount > 0) {
      return ranked.value.allDifficulty100 ? 'adversarial' : 'general';
    }
    return Number(difficulty.value) === 100 ? 'adversarial' : 'general';
  });

  let animIdCounter = 0;
  let transitionRevision = 0;
  let aiContinuationTimer = null;
  let aiRunning = false;
  let evilCoreModule = null;
  let aiWorker = null;
  let pendingAiMove = null;
  let evilGen = null;
  let lastAiSpeedRatio = null;
  let persistTimer = null;
  let rankedRng = null;
  let rankedStartSerial = 0;
  let rankedPollTimer = null;
  let rankedHeartbeatTimer = null;
  let rankedRunLock = null;
  let moveRunning = false;
  let gameGeneration = 0;
  let wasmPrewarmed = false;
  const history = [];

  const configuredSpawnRate4 = () => {
    const value = Number(appConfig.value?.['4_spawn_rate']);
    return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : SPAWN_RATE4;
  };

  const rankedSpawnRateAllowed = (value) => (
    Number(value) >= MIN_RANKED_SPAWN_RATE4 && Number(value) <= MAX_RANKED_SPAWN_RATE4
  );

  const writePersistedState = () => {
    gamerPreferencesStore.write({
      difficulty: difficulty.value,
      aiSpeed: aiSpeed.value,
      rankedParticipationEnabled: rankedParticipationEnabled.value,
      bestScore: Number(score.value.best) || 0,
    });
    gamerSessionStore.write({
      board: board.value.slice(0, 16),
      metadata: null,
      score: score.value,
      currentHex: currentHex.value,
      specialTiles: specialTiles.value,
      ranked: {
        ...ranked.value,
        rngState: rankedRng?.exportState?.() || ranked.value.rngState || null,
      },
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

  const clearRankedPollTimer = () => {
    if (rankedPollTimer !== null) {
      window.clearTimeout(rankedPollTimer);
      rankedPollTimer = null;
    }
  };

  const clearRankedHeartbeatTimer = () => {
    if (rankedHeartbeatTimer !== null) {
      window.clearTimeout(rankedHeartbeatTimer);
      rankedHeartbeatTimer = null;
    }
  };

  const updateRanked = (changes, { persist = true } = {}) => {
    ranked.value = { ...ranked.value, ...changes };
    if (persist) persistState({ immediate: !aiEnabled.value });
  };

  const disqualifyRanked = (reason) => {
    if (!ranked.value.runId || !ranked.value.eligible) return;
    updateRanked({ eligible: false, status: 'ineligible', errorCode: reason || '' });
  };

  const loseRankedOwnership = (reason = 'lease_lost') => {
    clearRankedHeartbeatTimer();
    rankedRunLock?.release();
    if (!ranked.value.runId) return;
    rankedRng = null;
    updateRanked({
      eligible: false,
      status: 'ineligible',
      errorCode: reason,
      leaseToken: '',
      leaseExpiresAt: null,
    });
  };

  rankedRunLock = createRankedRunLock({
    onLost: () => loseRankedOwnership('duplicate_tab'),
  });

  const hasRankedOwnership = () => Boolean(
    ranked.value.runId
    && ranked.value.leaseToken
    && rankedRunLock?.isHeld(ranked.value.runId)
  );

  const heartbeatCurrentRankedRun = async ({ immediateRetry = false } = {}) => {
    clearRankedHeartbeatTimer();
    const runId = ranked.value.runId;
    const leaseToken = ranked.value.leaseToken;
    if (!runId || !leaseToken || !rankedRunLock?.isHeld(runId)) return false;
    try {
      const payload = await heartbeatRankedRun(runId, leaseToken);
      if (runId !== ranked.value.runId || leaseToken !== ranked.value.leaseToken) return false;
      if (String(payload.status) !== 'active') {
        loseRankedOwnership(payload.error_code || 'lease_lost');
        return false;
      }
      updateRanked({ leaseExpiresAt: payload.lease_expires_at || null });
      rankedHeartbeatTimer = window.setTimeout(heartbeatCurrentRankedRun, 15000);
      return true;
    } catch (error) {
      if (runId !== ranked.value.runId || leaseToken !== ranked.value.leaseToken) return false;
      if ([401, 403, 404, 409].includes(Number(error?.status))) {
        loseRankedOwnership(error?.code || 'lease_lost');
        return false;
      }
      rankedHeartbeatTimer = window.setTimeout(
        heartbeatCurrentRankedRun,
        immediateRetry ? 2000 : 5000,
      );
      return true;
    }
  };

  const cancelRankedStart = () => {
    if (ranked.value.status !== 'starting') return;
    rankedStartSerial += 1;
    rankedRng = null;
    ranked.value = emptyRankedState({
      status: 'unranked',
      userId: authUser.value?.id || null,
    });
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
    const fromBoard = board.value.slice(0, 16);
    const toBoard = applySpecialTiles(values, specialTiles.value);
    board.value = toBoard;
    metadata.value = animation;
    transitionRevision += 1;
    transition.value = {
      id: transitionRevision,
      revision: transitionRevision,
      kind: animation ? 'move' : 'snapshot',
      fromBoard,
      toBoard: toBoard.slice(0, 16),
      metadata: animation,
    };
    score.value = {
      current: Number(nextScore) || 0,
      best: Math.max(Number(score.value.best) || 0, Number(nextScore) || 0),
    };
    syncDerivedState();
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

  const spawnEvil = async (values, { strict = false, spawnRate4 = configuredSpawnRate4() } = {}) => {
    const fallback = () => randomSpawn(values, spawnRate4);
    try {
      const module = evilCoreModule || await getEvilCore();
      evilCoreModule = module;
      if (!module?.EvilGen) {
        if (strict) throw new Error('EvilGen is unavailable.');
        return fallback();
      }
      const encoded = boardToEncoded(values);
      evilGen = evilGen || new module.EvilGen(encoded);
      evilGen.reset_board(encoded);
      const result = evilGen.gen_new_num(5);
      const index = Number(result?.[1]);
      const exponent = Number(result?.[2]);
      if (!Number.isInteger(index) || index < 0 || index >= 16 || exponent <= 0 || values[index] !== 0) {
        if (strict) throw new Error('EvilGen returned an invalid spawn.');
        return fallback();
      }
      return {
        index,
        value: exponentToValue(exponent),
      };
    } catch (error) {
      if (strict) throw error;
      console.error('EvilGen WASM spawn failed; falling back to random spawn.', error);
      return fallback();
    }
  };

  const rankedDeltaMs = () => {
    const now = Date.now();
    const delta = Math.max(0, Math.min(0xffffffff, now - Number(ranked.value.lastEventAt || now)));
    ranked.value.lastEventAt = now;
    return delta;
  };

  const appendRankedRecord = (record, byteCost) => {
    if (!ranked.value.runId || !ranked.value.eligible) return;
    const nextBytes = Number(ranked.value.byteEstimate || 40) + Number(byteCost || 0);
    const nextRecords = [...ranked.value.records, record];
    if (nextBytes > 500 * 1024) {
      disqualifyRanked('record_too_large');
      return;
    }
    ranked.value = {
      ...ranked.value,
      records: nextRecords,
      byteEstimate: nextBytes,
    };
  };

  const appendRankedMove = ({ direction, spawn, source }) => {
    if (!ranked.value.runId || !ranked.value.eligible) return;
    const currentDifficulty = Math.max(0, Math.min(100, Number(difficulty.value) || 0));
    if (ranked.value.lastDifficulty !== currentDifficulty) {
      appendRankedRecord([RANKED_RECORD.DIFFICULTY, currentDifficulty], 4);
      ranked.value.lastDifficulty = currentDifficulty;
    }
    const usedAi = String(source || '').includes('ai');
    if (usedAi && !ranked.value.usedAi) {
      appendRankedRecord([RANKED_RECORD.AI_USED], 3);
      ranked.value.usedAi = true;
    }
    const delta = rankedDeltaMs();
    appendRankedRecord([
      RANKED_RECORD.MOVE,
      DIRECTION_TO_NEXT_CODE[direction],
      spawn.index,
      spawn.value === 4 ? 1 : 0,
      delta,
    ], 1 + ulebLength(delta));
    const moveCount = ranked.value.moveCount + 1;
    ranked.value.moveCount = moveCount;
    ranked.value.allDifficulty100 = ranked.value.allDifficulty100 && currentDifficulty === 100;
    ranked.value.rngState = rankedRng?.exportState?.() || null;
    if (moveCount > MAX_RANKED_MOVES) disqualifyRanked('too_many_moves');
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

  const applyRankedServerStatus = (payload) => {
    if (!payload || payload.run_id !== ranked.value.runId) return;
    const status = String(payload.status || '');
    updateRanked({
      status,
      errorCode: payload.error_code || '',
      boardKey: payload.board_key || null,
      newPersonalBest: Boolean(payload.new_personal_best),
    });
  };

  const pollRankedRun = async () => {
    clearRankedPollTimer();
    const runId = ranked.value.runId;
    if (!runId || !['pending', 'validating'].includes(ranked.value.status)) return;
    try {
      const payload = await fetchRankedRun(runId);
      if (runId !== ranked.value.runId) return;
      applyRankedServerStatus(payload);
      if (['pending', 'validating'].includes(String(payload.status))) {
        rankedPollTimer = window.setTimeout(pollRankedRun, 2000);
      }
    } catch (_error) {
      if (runId === ranked.value.runId) {
        rankedPollTimer = window.setTimeout(pollRankedRun, 5000);
      }
    }
  };

  const submitCompletedRankedRun = async () => {
    if (!ranked.value.runId || !ranked.value.eligible) return;
    if (!hasRankedOwnership()) {
      loseRankedOwnership('lease_lost');
      return;
    }
    if (!ranked.value.records.some((record) => record?.[0] === RANKED_RECORD.END)) {
      appendRankedRecord([RANKED_RECORD.END], 1);
    }
    if (!ranked.value.eligible) return;
    updateRanked({ status: 'submitting', errorCode: '' });
    try {
      const recordEncoding = encodeRankedReplay({
        seedHex: ranked.value.seedHex,
        rulesVersion: ranked.value.rulesVersion,
        initialTiles: ranked.value.initialTiles,
        records: ranked.value.records,
      });
      const payload = await submitRankedRun(ranked.value.runId, {
        score: Math.floor(Number(score.value.current) || 0),
        final_board_codes: exactBoardCodes(board.value),
        record_encoding: recordEncoding,
        lease_token: ranked.value.leaseToken,
      });
      applyRankedServerStatus(payload);
      if (['pending', 'validating'].includes(String(payload.status))) {
        clearRankedHeartbeatTimer();
        rankedRunLock?.release();
        updateRanked({ leaseToken: '', leaseExpiresAt: null });
        pollRankedRun();
      }
    } catch (error) {
      if (ranked.value.runId) {
        if ([401, 403, 404, 409].includes(Number(error?.status))) {
          loseRankedOwnership(error?.code || 'lease_lost');
          return;
        }
        const errorCode = error?.code || 'submit_failed';
        const status = ['user_daily_limit', 'ip_daily_limit'].includes(errorCode)
          ? 'submission_limited'
          : 'submission_failed';
        updateRanked({ status, errorCode });
      }
    }
  };

  const retryRankedSubmission = () => {
    if (ranked.value.status === 'submission_failed') submitCompletedRankedRun();
  };

  const performMoveBoard = async (direction, source = 'manual') => {
    if (!VALID_DIRECTIONS.has(direction)) {
      return false;
    }
    cancelRankedStart();
    if (ranked.value.runId && ranked.value.eligible && !hasRankedOwnership()) {
      loseRankedOwnership('lease_lost');
    }
    const moveGeneration = gameGeneration;

    const before = board.value.slice(0, 16);
    const simulated = simulateMove(before, direction);
    if (boardsEqual(simulated.board, before)) {
      if (source === 'ai') {
        stopAI();
      }
      return false;
    }

    const nextScore = score.value.current + simulated.scoreDelta;
    let spawn = null;
    if (ranked.value.runId && ranked.value.eligible && rankedRng) {
      const currentDifficulty = Math.max(0, Math.min(100, Number(difficulty.value) || 0));
      const runSpawnRate4 = Number(ranked.value.spawnRate4 ?? SPAWN_RATE4);
      const branch = rankedRng.nextFloat();
      const useEvil = currentDifficulty >= 100
        || (currentDifficulty > 0 && branch < currentDifficulty / 100);
      if (useEvil) {
        try {
          spawn = await spawnEvil(simulated.board, { strict: true, spawnRate4: runSpawnRate4 });
        } catch (error) {
          console.error('Ranked EvilGen failed; this game is no longer rank eligible.', error);
          disqualifyRanked('evilgen_failed');
          spawn = randomSpawnWithRng(simulated.board, rankedRng, runSpawnRate4);
        }
      } else {
        spawn = randomSpawnWithRng(simulated.board, rankedRng, runSpawnRate4);
      }
    } else {
      const spawnRate4 = configuredSpawnRate4();
      spawn = Math.random() > (Number(difficulty.value) || 0) / 100
        ? randomSpawn(simulated.board, spawnRate4)
        : await spawnEvil(simulated.board, { spawnRate4 });
    }
    if (moveGeneration !== gameGeneration) return false;
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
    if (ranked.value.runId && ranked.value.eligible && spawn) {
      appendRankedMove({ direction, spawn, source });
    }
    persistState({ immediate: !aiEnabled.value });
    if (ranked.value.runId && ranked.value.eligible && !legalMoves(nextBoard).length) {
      submitCompletedRankedRun();
    }
    return true;
  };

  const moveBoard = async (direction, source = 'manual') => {
    if (moveRunning || ranked.value.status === 'starting') return false;
    moveRunning = true;
    try {
      return await performMoveBoard(direction, source);
    } finally {
      moveRunning = false;
    }
  };

  const initializeOrdinaryGame = (status = 'unranked', errorCode = '') => {
    rankedRng = null;
    ranked.value = emptyRankedState({ status, errorCode, userId: authUser.value?.id || null });
    const nextBoard = new Array(16).fill(0);
    const spawnRate4 = configuredSpawnRate4();
    const first = randomSpawn(nextBoard, spawnRate4);
    if (first) nextBoard[first.index] = first.value;
    const second = randomSpawn(nextBoard, spawnRate4);
    if (second) nextBoard[second.index] = second.value;
    return nextBoard;
  };

  const applyNewGameBoard = (nextBoard) => {
    stopAI();
    specialTiles.value = [];
    board.value = nextBoard;
    metadata.value = null;
    transition.value = null;
    score.value = {
      current: 0,
      best: Number(score.value.best) || 0,
    };
    syncDerivedState();
    history.length = 0;
    pushHistory();
    persistState({ immediate: true });
  };

  const newGame = async () => {
    gameGeneration += 1;
    const serial = ++rankedStartSerial;
    clearRankedPollTimer();
    stopAI();
    const replacement = ranked.value.runId && ranked.value.leaseToken
      ? { runId: ranked.value.runId, leaseToken: ranked.value.leaseToken }
      : {};
    if (!authUser.value?.id || !rankedParticipationEnabled.value) {
      clearRankedHeartbeatTimer();
      rankedRunLock?.release();
      if (replacement.runId) {
        abandonRankedRun(replacement.runId, replacement.leaseToken).catch(() => {});
      }
      applyNewGameBoard(initializeOrdinaryGame());
      return;
    }
    const requestedSpawnRate4 = configuredSpawnRate4();
    if (!rankedSpawnRateAllowed(requestedSpawnRate4)) {
      applyNewGameBoard(initializeOrdinaryGame('ineligible', 'spawn_rate_out_of_range'));
      return;
    }
    ranked.value = emptyRankedState({ status: 'starting', userId: authUser.value.id });
    persistState({ immediate: true });
    const leaseToken = createRankedLeaseToken();
    let issuedRunId = '';
    try {
      const run = await createRankedRun(
        createRankedRequestId(),
        requestedSpawnRate4,
        leaseToken,
        replacement,
      );
      issuedRunId = String(run.run_id || '');
      if (serial !== rankedStartSerial) {
        abandonRankedRun(run.run_id, leaseToken).catch(() => {});
        return;
      }
      if (String(run.status) !== 'active' || run.lease_token !== leaseToken) {
        throw Object.assign(new Error('Ranked lease was not issued.'), { code: 'lease_lost' });
      }
      const lockAcquired = await rankedRunLock.acquire(run.run_id);
      if (serial !== rankedStartSerial) {
        if (lockAcquired) rankedRunLock.release();
        abandonRankedRun(run.run_id, leaseToken).catch(() => {});
        return;
      }
      if (!lockAcquired) {
        abandonRankedRun(run.run_id, leaseToken).catch(() => {});
        applyNewGameBoard(initializeOrdinaryGame('ineligible', 'duplicate_tab'));
        return;
      }
      const runSpawnRate4 = Number(run.spawn_rate4);
      if (!rankedSpawnRateAllowed(runSpawnRate4)) throw new Error('Invalid ranked spawn rate.');
      const initialized = createRankedInitialBoard(run.seed_hex, runSpawnRate4);
      rankedRng = initialized.rng;
      ranked.value = emptyRankedState({
        runId: run.run_id,
        rulesVersion: run.rules_version,
        seedHex: run.seed_hex,
        rngState: initialized.rng.exportState(),
        initialTiles: initialized.initialTiles,
        eligible: true,
        status: 'ranked',
        startedAt: run.started_at,
        expiresAt: run.expires_at,
        userId: authUser.value.id,
        spawnRate4: runSpawnRate4,
        leaseToken,
        leaseExpiresAt: run.lease_expires_at || null,
        byteEstimate: 40,
      });
      applyNewGameBoard(initialized.board);
      rankedHeartbeatTimer = window.setTimeout(heartbeatCurrentRankedRun, 15000);
    } catch (error) {
      if (serial !== rankedStartSerial) return;
      clearRankedHeartbeatTimer();
      rankedRunLock?.release();
      if (issuedRunId) abandonRankedRun(issuedRunId, leaseToken).catch(() => {});
      applyNewGameBoard(initializeOrdinaryGame('ranked_unavailable', error?.code || 'start_failed'));
    }
  };

  const undo = () => {
    if (history.length <= 1) {
      return false;
    }
    cancelRankedStart();
    gameGeneration += 1;
    stopAI();
    history.pop();
    if (ranked.value.runId && ranked.value.eligible) {
      const delta = rankedDeltaMs();
      appendRankedRecord([RANKED_RECORD.UNDO, 1, delta], 1 + ulebLength(delta));
      updateRanked({ usedUndo: true });
      disqualifyRanked('undo_used');
    }
    const previous = history[history.length - 1];
    specialTiles.value = previous.specialTiles;
    board.value = applySpecialTiles(previous.board, previous.specialTiles);
    metadata.value = null;
    transition.value = null;
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
    cancelRankedStart();
    gameGeneration += 1;
    stopAI();
    if (ranked.value.runId && ranked.value.eligible) {
      updateRanked({ usedSetBoard: true });
      disqualifyRanked('set_board_used');
    }
    specialTiles.value = [];
    board.value = boardFromHex(normalized);
    metadata.value = null;
    transition.value = null;
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

  const setRankedParticipationEnabled = (enabled) => {
    rankedParticipationEnabled.value = Boolean(enabled);
    persistState({ immediate: true });
  };

  const openBrowserAi = () => false;

  const loadSavedState = async () => {
    const legacy = legacyGamerStore.read();
    let preferences = gamerPreferencesStore.read();
    if (!preferences && legacy) {
      preferences = {
        difficulty: legacy.difficulty,
        aiSpeed: legacy.aiSpeed,
        rankedParticipationEnabled: legacy.rankedParticipationEnabled,
        bestScore: legacy.score?.best,
      };
      gamerPreferencesStore.write(preferences);
    }
    legacyGamerStore.remove();
    rankedParticipationEnabled.value = preferences?.rankedParticipationEnabled !== false;
    difficulty.value = Math.max(0, Math.min(100, Number(preferences?.difficulty) || 0));
    aiSpeed.value = Math.max(0, Math.min(200, Number(preferences?.aiSpeed) || 100));

    const saved = gamerSessionStore.read();
    if (!saved?.board || !Array.isArray(saved.board) || saved.board.length !== 16) {
      await newGame();
      return;
    }
    specialTiles.value = Array.isArray(saved.specialTiles) ? saved.specialTiles : [];
    board.value = applySpecialTiles(saved.board, specialTiles.value);
    metadata.value = null;
    transition.value = null;
    score.value = {
      current: Number(saved.score?.current) || 0,
      best: Math.max(Number(saved.score?.best) || 0, Number(preferences?.bestScore) || 0),
    };
    syncDerivedState();
    history.length = 0;
    pushHistory();

    const savedRanked = saved.ranked;
    if (!savedRanked?.runId || !savedRanked?.seedHex || !Array.isArray(savedRanked?.records)) {
      rankedRng = null;
      ranked.value = emptyRankedState({
        status: 'legacy_unranked',
        userId: authUser.value?.id || null,
      });
      return;
    }
    try {
      const restoredRanked = emptyRankedState({
        ...savedRanked,
        spawnRate4: Number(savedRanked.spawnRate4 ?? SPAWN_RATE4),
        records: savedRanked.records.map((record) => [...record]),
        initialTiles: savedRanked.initialTiles.map((tile) => [...tile]),
      });
      rankedRng = new Xoshiro128StarStar(restoredRanked.rngState);
      ranked.value = restoredRanked;
      if (
        authReady.value
        && Number(ranked.value.userId) !== Number(authUser.value?.id || 0)
        && ranked.value.eligible
        && ranked.value.status === 'ranked'
      ) {
        loseRankedOwnership('user_changed');
        return;
      }
      if (['pending', 'validating'].includes(ranked.value.status)) pollRankedRun();
      const needsLease = ['ranked', 'ineligible', 'submitting', 'submission_failed']
        .includes(ranked.value.status);
      if (needsLease) {
        if (!ranked.value.leaseToken) {
          applyNewGameBoard(initializeOrdinaryGame('ineligible', 'lease_required'));
          return;
        }
        const originalStatus = ranked.value.status === 'submitting'
          ? 'submission_failed'
          : ranked.value.status;
        const originalEligible = ranked.value.eligible;
        const restoringRunId = ranked.value.runId;
        updateRanked({ status: 'starting', eligible: false });
        const acquired = await rankedRunLock.acquire(restoringRunId);
        if (ranked.value.runId !== restoringRunId) {
          if (acquired) rankedRunLock.release();
          return;
        }
        if (!acquired) {
          applyNewGameBoard(initializeOrdinaryGame('ineligible', 'duplicate_tab'));
          return;
        }
        updateRanked({ status: originalStatus, eligible: originalEligible });
        if (!await heartbeatCurrentRankedRun({ immediateRetry: true })) return;
        if (originalStatus === 'submission_failed') {
          window.setTimeout(submitCompletedRankedRun, 0);
        }
      }
    } catch (_error) {
      clearRankedHeartbeatTimer();
      rankedRunLock?.release();
      rankedRng = null;
      ranked.value = emptyRankedState({ status: 'legacy_unranked' });
    }
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

  const ensureWasmPrewarmed = () => {
    if (wasmPrewarmed) return;
    wasmPrewarmed = true;
    prewarmWasmEngines();
  };

  const handleKeydown = (event) => {
    if (!activeRef?.value || !keyboardInputAllowed(KEYBOARD_OWNERS.PRIMARY)) return;
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

  const handleVisibilityChange = () => {
    if (document.hidden) {
      flushPersistState();
    } else if (hasRankedOwnership()) {
      void heartbeatCurrentRankedRun({ immediateRetry: true });
    }
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
      if (isActive) {
        ensureWasmPrewarmed();
      }
      if (isActive && aiEnabled.value) {
        scheduleAiStep();
      } else {
        clearAiContinuationTimer();
        if (!isActive) flushPersistState();
      }
    },
    { immediate: true }
  );

  watch(
    [authReady, () => authUser.value?.id],
    ([ready, userId]) => {
      if (!ready || !ranked.value.runId || !ranked.value.eligible || ranked.value.status !== 'ranked') return;
      if (Number(ranked.value.userId) !== Number(userId || 0)) loseRankedOwnership('user_changed');
    },
  );

  onMounted(() => {
    void loadSavedState();
    if (activeRef?.value) {
      ensureWasmPrewarmed();
    }
    window.addEventListener('keydown', handleKeydown);
    window.addEventListener('beforeunload', handleBeforeUnload);
    document.addEventListener('visibilitychange', handleVisibilityChange);
  });

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown);
    window.removeEventListener('beforeunload', handleBeforeUnload);
    document.removeEventListener('visibilitychange', handleVisibilityChange);
    clearRankedPollTimer();
    clearRankedHeartbeatTimer();
    rankedRunLock?.release();
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
    transition,
    score,
    wsStatus,
    aiEnabled,
    difficulty,
    aiSpeed,
    hexInput,
    scoreAnimations,
    aiWorkerReady,
    rankedParticipationEnabled,
    rankedStatus,
    rankedMode,
    ranked,
    triggerAction,
    toggleAI,
    updateSettings,
    setBoard,
    writeCurrentBoardToHex,
    setRankedParticipationEnabled,
    retryRankedSubmission,
    openBrowserAi,
  };
}
