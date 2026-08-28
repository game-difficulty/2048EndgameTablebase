import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { KEYBOARD_OWNERS, keyboardInputAllowed } from '../../../app/keyboardOwnership';
import { useAuthState } from '../../../services/auth/authState';
import { createExclusiveRunLock } from '../../../services/concurrency/exclusiveRunLock';
import { createLocalStorageStore } from '../../../services/storage/localStorageStore';
import { MinigameController } from '../engine/controller';
import { MinigameRankedRecorder } from '../engine/rankedRecorder';
import { createMinigameRuntime, restoreMinigameRuntime } from '../engine/runtime';
import { MGO1_END_REASON } from '../protocol';
import { createEmptyMinigameMenu, createEmptyMinigameState } from '../model/minigameViewState';
import {
  abandonMinigameRankedRun,
  claimMinigameRankedRun,
  createMinigameRankedRun,
  createMinigameRequestId,
  fetchMinigameRankedRun,
  heartbeatMinigameRankedRun,
  qualifyMinigameRankedRun,
  submitMinigameRankedRun,
} from '../services/minigameRankingClient';
import {
  readMinigameLease,
  removeMinigameLease,
  writeMinigameLease,
} from '../services/minigameLeaseStore';

const isTextEntryElement = (element) => {
  if (!(element instanceof HTMLElement)) {
    return false;
  }
  const tagName = element.tagName;
  return tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT' || element.isContentEditable;
};

const normalizeHudPanels = (hud, receivedAt = Date.now()) => {
  const customPanels = Array.isArray(hud?.customPanels) ? hud.customPanels : [];
  return {
    ...(hud || {}),
    customPanels: customPanels.map((panel) => {
      if (panel?.type !== 'countdown') return panel;
      return {
        ...panel,
        syncedAt: receivedAt,
      };
    }),
  };
};

const defaultMinigameState = () => ({
  difficulty: 1,
  summaries: {},
  activeGameSnapshots: {},
});

const minigameStore = createLocalStorageStore({
  key: 'minigames',
  version: 3,
  defaultValue: defaultMinigameState(),
  migrate(value) {
    const previous = value && typeof value === 'object' ? value : {};
    return {
      ...defaultMinigameState(),
      difficulty: Number(previous.difficulty) ? 1 : 0,
      summaries: previous.summaries && typeof previous.summaries === 'object' ? previous.summaries : {},
      activeGameSnapshots: previous.activeGameSnapshots && typeof previous.activeGameSnapshots === 'object'
        ? previous.activeGameSnapshots
        : {},
    };
  },
});

const normalizeStoredState = () => ({
  ...defaultMinigameState(),
  ...(minigameStore.read() || {}),
});

const snapshotKey = (gameId, difficulty) => `${gameId || ''}:${Number(difficulty) ? 1 : 0}`;
const LEASE_FREE_RANKED_STATES = new Set([
  'pending', 'validating', 'verified', 'no_improvement', 'not_candidate', 'rejected', 'expired',
]);

export function useMinigameSession(activeRef) {
  const { t } = useI18n();
  const { user: authUser } = useAuthState();

  const storedState = ref(normalizeStoredState());
  const menuData = ref({
    ...createEmptyMinigameMenu(),
    difficulty: Number(storedState.value.difficulty) ? 1 : 0,
  });
  const gameState = ref(createEmptyMinigameState());
  const lastMenuFocusGameId = ref('');
  const toastMessage = ref('');
  const inputLockedUntil = ref(0);
  const overlay = ref({
    open: false,
    type: 'info',
    title: '',
    message: '',
    level: '',
  });
  const pendingOverlay = ref(null);
  const rankedStatus = ref('unranked');

  let controller = null;
  let activeRecorder = null;
  let toastTimer = null;
  let inputLockTimer = null;
  let rankedPollTimer = null;
  let rankedHeartbeatTimer = null;
  let rankedHeartbeatInFlight = false;
  let activeLeaseToken = '';
  let rankedPersistenceBlocked = false;
  let timedTickTimer = null;
  let timedTickInFlight = false;
  const submittedFinals = new Set();

  const hasRankedOwnership = () => Boolean(
    activeRecorder
    && activeLeaseToken
    && rankedRunLock.isHeld(activeRecorder.runId)
  );

  const stopRankedHeartbeat = () => {
    if (rankedHeartbeatTimer) window.clearInterval(rankedHeartbeatTimer);
    rankedHeartbeatTimer = null;
    rankedHeartbeatInFlight = false;
  };

  const releaseRankedOwnership = ({ forgetLease = false } = {}) => {
    const runId = activeRecorder?.runId || null;
    stopRankedHeartbeat();
    const releasedRunId = rankedRunLock.release();
    activeLeaseToken = '';
    if (forgetLease && (runId || releasedRunId)) removeMinigameLease(runId || releasedRunId);
  };

  const loseRankedOwnership = (runId) => {
    if (runId) removeMinigameLease(runId);
    if (activeRecorder?.runId !== runId) return;
    stopRankedHeartbeat();
    activeLeaseToken = '';
    rankedPersistenceBlocked = true;
    if (activeRecorder?.runId === runId) {
      activeRecorder.submissionState = 'invalid';
      activeRecorder = null;
      rankedStatus.value = 'invalid';
    }
  };

  const rankedRunLock = createExclusiveRunLock({
    namespace: 'minigame-ranked',
    onLost: loseRankedOwnership,
  });

  const heartbeatActiveRankedRun = async () => {
    if (!hasRankedOwnership() || rankedHeartbeatInFlight) return false;
    const recorder = activeRecorder;
    const leaseToken = activeLeaseToken;
    rankedHeartbeatInFlight = true;
    try {
      const result = await heartbeatMinigameRankedRun(recorder.runId, leaseToken);
      writeMinigameLease(recorder.runId, {
        leaseToken,
        expiresAt: result?.lease_expires_at,
        userId: recorder.userId,
      });
      return true;
    } catch (error) {
      if ([403, 404, 409, 410].includes(Number(error?.status))) {
        loseRankedOwnership(recorder.runId);
      }
      return false;
    } finally {
      rankedHeartbeatInFlight = false;
    }
  };

  const startRankedHeartbeat = () => {
    stopRankedHeartbeat();
    rankedHeartbeatTimer = window.setInterval(() => {
      void heartbeatActiveRankedRun();
    }, 15_000);
  };

  const submitFinishedGame = async (state) => {
    const snapshot = state?.snapshot;
    const engine = snapshot?.engine;
    const recorder = activeRecorder;
    if (!authUser.value || !snapshot?.gameId || !engine || !recorder?.ended) return;
    if (!hasRankedOwnership()) return;
    if (Number(recorder.userId) !== Number(authUser.value.id)) return;
    if (['invalid', 'too_large', 'not_candidate', 'pending', 'verified', 'no_improvement'].includes(recorder.submissionState)) return;
    const board = Array.isArray(state.board) ? state.board.map((value) => Number(value)) : [];
    const rows = Number(state.shape?.rows || 0);
    const cols = Number(state.shape?.cols || 0);
    if (!board.length || rows * cols !== board.length) return;
    const fingerprint = recorder.runId;
    if (submittedFinals.has(fingerprint)) return;
    submittedFinals.add(fingerprint);
    try {
      recorder.submissionState = 'qualifying';
      rankedStatus.value = 'qualifying';
      persistRecorderState();
      const qualification = await qualifyMinigameRankedRun(recorder.runId, {
        run_token: recorder.runToken,
        lease_token: activeLeaseToken,
        score: Math.max(0, Math.trunc(Number(state.score || 0))),
        trophy_tier: Math.max(0, Math.min(4, Math.trunc(Number(engine.isPassed || 0)))),
        highest_tile_exp: Math.max(0, Math.min(63, Math.trunc(Number(engine.highestTileExp || engine.maxNum || 0)))),
        final_board: board.map((value) => Math.trunc(value)),
        board_rows: rows,
        board_cols: cols,
        action_count: recorder.mutableActionCount,
        elapsed_ms: recorder.elapsedMs,
      });
      if (!qualification?.candidate) {
        recorder.submissionState = 'not_candidate';
        rankedStatus.value = 'not_candidate';
        persistRecorderState();
        releaseRankedOwnership({ forgetLease: true });
        return;
      }
      if (!qualification?.submission_token) {
        const recoveredStatus = String(qualification?.status || 'submit_failed');
        recorder.submissionState = recoveredStatus;
        rankedStatus.value = recoveredStatus;
        persistRecorderState();
        if (['pending', 'validating'].includes(recoveredStatus)) {
          releaseRankedOwnership({ forgetLease: true });
          scheduleRankedStatusPoll(recorder.runId);
        } else if (recoveredStatus === 'verified') {
          releaseRankedOwnership({ forgetLease: true });
          window.dispatchEvent(new CustomEvent('minigame-score-updated', {
            detail: {
              gameId: snapshot.gameId,
              difficulty: Number(snapshot.difficulty) ? 1 : 0,
              result: qualification,
            },
          }));
        }
        return;
      }
      recorder.submissionState = 'submitting';
      rankedStatus.value = 'submitting';
      persistRecorderState();
      const result = await submitMinigameRankedRun(recorder.runId, {
        submission_token: qualification.submission_token,
        lease_token: activeLeaseToken,
        record_encoding: recorder.encode(),
      });
      recorder.submissionState = String(result?.status || 'pending');
      rankedStatus.value = recorder.submissionState;
      persistRecorderState();
      releaseRankedOwnership({ forgetLease: true });
      scheduleRankedStatusPoll(recorder.runId);
      window.dispatchEvent(new CustomEvent('minigame-score-updated', {
        detail: {
          gameId: snapshot.gameId,
          difficulty: Number(snapshot.difficulty) ? 1 : 0,
          result,
        },
      }));
    } catch (error) {
      if (error?.status !== 401) {
        submittedFinals.delete(fingerprint);
        recorder.submissionState = 'submit_failed';
        rankedStatus.value = 'submit_failed';
        persistRecorderState();
        if ([403, 404, 409, 410].includes(Number(error?.status))) {
          loseRankedOwnership(recorder.runId);
        }
        console.warn('Ranked minigame submission failed.', error);
      }
    }
  };

  const hasActiveGame = computed(() => Boolean(gameState.value?.gameId));
  const currentView = computed(() => (hasActiveGame.value ? 'play' : 'menu'));
  const menuSections = computed(() => menuData.value.sections || []);
  const difficulty = computed(() => Number(menuData.value.difficulty ?? 1));
  const recordOperation = ({ operation, atMs, state }) => {
    if (!activeRecorder || activeRecorder.ended || !hasRankedOwnership()) return;
    try {
      const recorded = activeRecorder.record(operation, atMs, state);
      if (!recorded) {
        rankedStatus.value = activeRecorder.submissionState;
        return;
      }
      if (state?.engine?.isOver) {
        const finished = activeRecorder.finish(atMs, state);
        rankedStatus.value = finished ? 'finished' : activeRecorder.submissionState;
      }
    } catch (error) {
      activeRecorder.submissionState = 'invalid';
      rankedStatus.value = 'invalid';
      console.warn('Ranked minigame recording stopped.', error);
    }
  };
  const ensureController = () => {
    if (!controller) {
      controller = new MinigameController({
        difficulty: Number(storedState.value.difficulty) ? 1 : 0,
        summaries: storedState.value.summaries || {},
        snapshotKey,
        onOperation: recordOperation,
      });
    }
    controller.setDifficulty(Number(storedState.value.difficulty) ? 1 : 0);
    controller.setSummaries(storedState.value.summaries || {});
    return controller;
  };

  const refreshMenu = () => {
    menuData.value = {
      ...createEmptyMinigameMenu(),
      ...ensureController().menuPayload(),
    };
  };

  const showToast = (message) => {
    toastMessage.value = message;
    if (toastTimer) window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
      toastMessage.value = '';
      toastTimer = null;
    }, 2200);
  };

  const lockInputFor = (ms) => {
    const duration = Math.max(0, Number(ms || 0));
    if (!duration) return;
    const nextLockedUntil = Date.now() + duration;
    inputLockedUntil.value = Math.max(inputLockedUntil.value, nextLockedUntil);
    if (inputLockTimer) {
      window.clearTimeout(inputLockTimer);
    }
    inputLockTimer = window.setTimeout(() => {
      inputLockedUntil.value = 0;
      inputLockTimer = null;
    }, Math.max(0, inputLockedUntil.value - Date.now()));
  };

  const openOverlay = (payload) => {
    overlay.value = {
      open: true,
      type: payload.type || 'info',
      title: payload.title || '',
      message: payload.message || '',
      level: payload.level || '',
    };
  };

  const closeOverlay = () => {
    if (pendingOverlay.value) {
      const nextOverlay = pendingOverlay.value;
      pendingOverlay.value = null;
      overlay.value = {
        open: true,
        type: nextOverlay.type || 'info',
        title: nextOverlay.title || '',
        message: nextOverlay.message || '',
        level: nextOverlay.level || '',
      };
      return;
    }
    overlay.value = {
      open: false,
      type: 'info',
      title: '',
      message: '',
      level: '',
    };
  };

  const persistState = (updater) => {
    storedState.value = minigameStore.update((current) => {
      const base = {
        ...defaultMinigameState(),
        ...(current || {}),
      };
      return updater(base);
    });
    refreshMenu();
  };

  const persistRecorderState = () => {
    const gameId = String(gameState.value?.gameId || '');
    if (!gameId || !activeRecorder) return;
    if (rankedPersistenceBlocked) return;
    if (!hasRankedOwnership() && !LEASE_FREE_RANKED_STATES.has(activeRecorder.submissionState)) return;
    const key = snapshotKey(gameId, gameState.value?.snapshot?.difficulty ?? difficulty.value);
    storedState.value = minigameStore.update((current) => {
      const existing = current?.activeGameSnapshots?.[key];
      if (!existing) return current;
      return {
        ...current,
        activeGameSnapshots: {
          ...(current.activeGameSnapshots || {}),
          [key]: {
            ...existing,
            rankedRun: activeRecorder.exportSnapshot(),
          },
        },
      };
    });
  };

  const scheduleRankedStatusPoll = (runId, attempt = 0, eventContext = null) => {
    if (rankedPollTimer) window.clearTimeout(rankedPollTimer);
    if (!runId || attempt >= 30) return;
    const context = eventContext || {
      gameId: gameState.value?.gameId,
      difficulty: Number(gameState.value?.snapshot?.difficulty) ? 1 : 0,
    };
    rankedPollTimer = window.setTimeout(async () => {
      rankedPollTimer = null;
      try {
        const result = await fetchMinigameRankedRun(runId);
        const status = String(result?.status || '');
        if (activeRecorder?.runId === runId) {
          activeRecorder.submissionState = status || activeRecorder.submissionState;
          rankedStatus.value = activeRecorder.submissionState;
          persistRecorderState();
        }
        if (['pending', 'validating'].includes(status)) {
          scheduleRankedStatusPoll(runId, attempt + 1, context);
        } else if (status === 'verified') {
          window.dispatchEvent(new CustomEvent('minigame-score-updated', {
            detail: {
              gameId: context.gameId,
              difficulty: context.difficulty,
              result,
            },
          }));
        }
      } catch (error) {
        if (error?.status !== 401) scheduleRankedStatusPoll(runId, attempt + 1, context);
      }
    }, Math.min(10_000, 1500 + attempt * 500));
  };

  const handleStateData = (payload) => {
    const recorderSnapshot = activeRecorder
      && !rankedPersistenceBlocked
      && (hasRankedOwnership() || LEASE_FREE_RANKED_STATES.has(activeRecorder.submissionState))
      ? activeRecorder.exportSnapshot()
      : null;
    const effectivePayload = payload?.snapshot
      ? {
        ...payload,
        snapshot: {
          ...payload.snapshot,
          rankedRun: recorderSnapshot,
        },
      }
      : payload;
    const previousStatus = String(gameState.value?.status || '');
    const receivedAt = Date.now();
    const nextState = {
      ...createEmptyMinigameState(),
      ...(effectivePayload || {}),
      shape: {
        ...createEmptyMinigameState().shape,
        ...(effectivePayload?.shape || {}),
      },
      view: {
        ...createEmptyMinigameState().view,
        ...(effectivePayload?.view || {}),
      },
      hud: {
        ...createEmptyMinigameState().hud,
        ...normalizeHudPanels(effectivePayload?.hud || {}, receivedAt),
      },
      powerups: {
        ...createEmptyMinigameState().powerups,
        ...(effectivePayload?.powerups || {}),
      },
      interaction: {
        ...createEmptyMinigameState().interaction,
        ...(effectivePayload?.interaction || {}),
      },
      messages: effectivePayload?.messages || {},
    };
    gameState.value = nextState;
    void submitFinishedGame(nextState);
    const snapshot = nextState.snapshot;
    if (snapshot?.gameId) {
      const key = snapshotKey(snapshot.gameId, snapshot.difficulty ?? difficulty.value);
      const boardValues = Array.isArray(nextState.board) ? nextState.board : [];
      const highestExp = boardValues.reduce((maxExp, value) => {
        const numeric = Number(value || 0);
        if (numeric <= 0) return maxExp;
        return Math.max(maxExp, numeric);
      }, 0);
      persistState((current) => {
        const previousSummary = current.summaries?.[key] || {};
        const engineHighestExp = Math.max(
          Number(snapshot?.engine?.maxNum || 0),
          Number(snapshot?.engine?.highestTileExp || 0)
        );
        const summaryHighestExp = Math.max(
          Number(previousSummary.highestExp || 0),
          Number(highestExp || 0),
          engineHighestExp
        );
        const trophy = Math.max(
          Number(previousSummary.trophy || 0),
          Number(snapshot?.engine?.isPassed || 0)
        );
        const nextStoredState = {
          ...current,
          difficulty: Number(snapshot.difficulty) ? 1 : 0,
          summaries: {
            ...(current.summaries || {}),
            [key]: {
              bestScore: Math.max(Number(previousSummary.bestScore || 0), Number(nextState.best || 0)),
              highestTile: summaryHighestExp > 0 ? 2 ** summaryHighestExp : 0,
              highestExp: summaryHighestExp,
              trophy,
            },
          },
        };
        if (rankedPersistenceBlocked) return nextStoredState;
        return {
          ...nextStoredState,
          activeGameSnapshots: {
            ...(current.activeGameSnapshots || {}),
            [key]: snapshot,
          },
        };
      });
    }

    const animation = gameState.value.animation || {};
    const followUp = animation.followUp || null;
    if (followUp) {
      if (followUp.lockInput !== false) {
        lockInputFor(Number(followUp.delayMs || 0) + Number(followUp.durationMs || 0));
      }
    } else {
      const effectDurations = []
        .concat(Array.isArray(animation.effects) ? animation.effects : [])
        .concat(Array.isArray(animation.pageEffects) ? animation.pageEffects : []);
      if (effectDurations.length) {
        const maxEffectDuration = effectDurations.reduce(
          (duration, effect) =>
            Math.max(
              duration,
              Number(effect?.delayMs || 0) + Number(effect?.durationMs || effect?.animDurationMs || 430)
            ),
          430
        );
        lockInputFor(maxEffectDuration);
      }
    }

    const messages = gameState.value.messages || {};
    const enteredGameOver = nextState.status === 'game_over' && previousStatus !== 'game_over';
    pendingOverlay.value = null;
    if (messages.toast) {
      showToast(messages.toast);
    }
    if (messages.infoDialog) {
      openOverlay({
        type: 'info',
        title: t('minigames.overlay.infoTitle'),
        message: String(messages.infoDialog),
      });
    } else if (messages.trophy) {
      if (messages.gameOver || enteredGameOver) {
        pendingOverlay.value = {
          type: 'gameOver',
          title: t('minigames.overlay.gameOverTitle'),
          message: String(messages.gameOver || t('minigames.overlay.gameOverMessage')),
          level: '',
        };
      }
      openOverlay({
        type: 'trophy',
        title: t('minigames.overlay.trophyTitle'),
        message: messages.trophy.message || '',
        level: messages.trophy.level || '',
      });
    } else if (messages.gameOver || enteredGameOver) {
      openOverlay({
        type: 'gameOver',
        title: t('minigames.overlay.gameOverTitle'),
        message: String(messages.gameOver || t('minigames.overlay.gameOverMessage')),
      });
    } else if (nextState.status !== 'game_over' && overlay.value.type === 'gameOver' && overlay.value.open) {
      closeOverlay();
    }
  };

  const runLocalAction = async (callback) => {
    try {
      const payload = await callback(ensureController());
      if (payload?.gameId) {
        handleStateData(payload);
      } else {
        refreshMenu();
      }
      return true;
    } catch (error) {
      console.error('Minigame local action failed.', error);
      showToast(error?.message || 'Minigame action failed.');
      return false;
    }
  };

  const setDifficulty = (value) => {
    const nextDifficulty = Number(value) ? 1 : 0;
    persistState((current) => ({
      ...current,
      difficulty: nextDifficulty,
    }));
    menuData.value = {
      ...menuData.value,
      difficulty: nextDifficulty,
    };
    ensureController().setDifficulty(nextDifficulty);
    refreshMenu();
  };

  const restoreRankedRecorder = (snapshot, gameId, selectedDifficulty) => {
    const restored = MinigameRankedRecorder.restore(snapshot?.rankedRun);
    const expiresAtMs = Date.parse(String(restored?.expiresAt || ''));
    if (
      !restored
      || !authUser.value
      || Number(restored.userId) !== Number(authUser.value.id)
      || restored.gameId !== gameId
      || restored.difficulty !== (Number(selectedDifficulty) ? 1 : 0)
      || (Number.isFinite(expiresAtMs) && expiresAtMs <= Date.now())
    ) {
      return null;
    }
    if (!restored.ended) restored.lastActionAtMs = Date.now();
    return restored;
  };

  const markRankedRunInvalid = () => {
    if (!activeRecorder?.ended) {
      const runId = activeRecorder.runId;
      const leaseToken = activeLeaseToken;
      activeRecorder.submissionState = 'invalid';
      rankedStatus.value = 'invalid';
      persistRecorderState();
      if (runId && leaseToken) {
        void abandonMinigameRankedRun(runId, leaseToken).catch(() => {});
      }
      releaseRankedOwnership({ forgetLease: true });
    }
  };

  const activateRankedRecorder = (recorder, leaseToken, leaseExpiresAt = '') => {
    activeRecorder = recorder;
    activeLeaseToken = String(leaseToken || '');
    rankedPersistenceBlocked = false;
    rankedStatus.value = recorder.submissionState || 'active';
    writeMinigameLease(recorder.runId, {
      leaseToken: activeLeaseToken,
      expiresAt: leaseExpiresAt,
      userId: recorder.userId,
    });
    startRankedHeartbeat();
  };

  const restoreRankedOwnership = async (recorder) => {
    if (!recorder) return null;
    if (LEASE_FREE_RANKED_STATES.has(recorder.submissionState)) {
      activeRecorder = recorder;
      rankedPersistenceBlocked = false;
      rankedStatus.value = recorder.submissionState;
      return recorder;
    }
    if (!await rankedRunLock.acquire(recorder.runId)) {
      rankedPersistenceBlocked = true;
      rankedStatus.value = 'unranked';
      return null;
    }
    const storedLease = readMinigameLease(recorder.runId);
    let leaseToken = Number(storedLease?.userId) === Number(recorder.userId)
      ? String(storedLease?.leaseToken || '')
      : '';
    try {
      let result;
      if (leaseToken) {
        result = await heartbeatMinigameRankedRun(recorder.runId, leaseToken);
      } else {
        leaseToken = createMinigameRequestId();
        result = await claimMinigameRankedRun(recorder.runId, leaseToken);
        if (result?.run_token) recorder.runToken = result.run_token;
      }
      activateRankedRecorder(recorder, leaseToken, result?.lease_expires_at);
      return recorder;
    } catch (error) {
      rankedRunLock.release();
      removeMinigameLease(recorder.runId);
      rankedPersistenceBlocked = true;
      rankedStatus.value = 'unranked';
      if (![401, 403, 404, 409, 410].includes(Number(error?.status))) {
        console.warn('Unable to restore ranked minigame ownership.', error);
      }
      return null;
    }
  };

  const createRankedContext = async (gameId, selectedDifficulty) => {
    releaseRankedOwnership();
    activeRecorder = null;
    rankedStatus.value = 'unranked';
    rankedPersistenceBlocked = false;
    if (!authUser.value) return createMinigameRuntime();
    const leaseToken = createMinigameRequestId();
    try {
      const run = await createMinigameRankedRun({
        requestId: createMinigameRequestId(),
        gameId,
        difficulty: selectedDifficulty,
        leaseToken,
      });
      if (!await rankedRunLock.acquire(run.run_id)) {
        await abandonMinigameRankedRun(run.run_id, leaseToken).catch(() => {});
        rankedPersistenceBlocked = true;
        return createMinigameRuntime();
      }
      const runtime = createMinigameRuntime({
        seedHex: run.seed_hex,
        onDeterminismFailure: markRankedRunInvalid,
      });
      const recorder = new MinigameRankedRecorder({
        runId: run.run_id,
        userId: authUser.value.id,
        gameId,
        difficulty: selectedDifficulty,
        seedHex: run.seed_hex,
        rulesVersion: run.rules_version,
        runToken: run.run_token,
        expiresAt: run.expires_at,
        startedAtMs: Date.now(),
      });
      activateRankedRecorder(recorder, leaseToken, run.lease_expires_at);
      return runtime;
    } catch (error) {
      rankedRunLock.release();
      if (Number(error?.status) === 409) rankedPersistenceBlocked = true;
      if (error?.status !== 401) console.warn('Unable to create ranked minigame run.', error);
      return createMinigameRuntime();
    }
  };

  const startGame = async (gameId) => {
    lastMenuFocusGameId.value = String(gameId || '');
    closeOverlay();
    const key = snapshotKey(gameId, difficulty.value);
    const snapshot = storedState.value.activeGameSnapshots?.[key] || null;
    releaseRankedOwnership();
    activeRecorder = null;
    rankedPersistenceBlocked = false;
    const restoredRecorder = restoreRankedRecorder(snapshot, String(gameId || ''), difficulty.value);
    await restoreRankedOwnership(restoredRecorder);
    let runtime = null;
    if (!snapshot) runtime = await createRankedContext(String(gameId || ''), difficulty.value);
    else if (activeRecorder && snapshot?.runtime) {
      runtime = restoreMinigameRuntime(snapshot.runtime, {
        onDeterminismFailure: markRankedRunInvalid,
      });
    }
    await runLocalAction((localController) => localController.startGame(gameId, snapshot, runtime));
    if (['pending', 'validating'].includes(activeRecorder?.submissionState)) {
      scheduleRankedStatusPoll(activeRecorder.runId);
    } else if (activeRecorder?.ended) {
      void submitFinishedGame(gameState.value);
    }
  };

  const backToMenu = async () => {
    lastMenuFocusGameId.value = String(gameState.value?.gameId || lastMenuFocusGameId.value || '');
    closeOverlay();
    const recorder = activeRecorder;
    const leaseToken = activeLeaseToken;
    const finalState = gameState.value;
    const trophyTier = Math.max(
      0,
      Math.min(4, Math.trunc(Number(finalState?.snapshot?.engine?.isPassed || 0))),
    );
    if (recorder && !recorder.ended && hasRankedOwnership() && trophyTier > 0) {
      const finished = recorder.finish(Date.now(), null, MGO1_END_REASON.RETIRED);
      rankedStatus.value = finished ? 'finished' : recorder.submissionState;
      persistRecorderState();
      if (finished) await submitFinishedGame(finalState);
    } else if (recorder?.runId && leaseToken && hasRankedOwnership()) {
      await abandonMinigameRankedRun(recorder.runId, leaseToken).catch(() => {});
    }
    ensureController().backToMenu();
    releaseRankedOwnership();
    activeRecorder = null;
    rankedStatus.value = 'unranked';
    gameState.value = createEmptyMinigameState();
    refreshMenu();
  };

  const newGame = async () => {
    closeOverlay();
    const previousRunId = activeRecorder?.runId || '';
    const previousLeaseToken = activeLeaseToken;
    const ownedPreviousRun = hasRankedOwnership();
    if (ownedPreviousRun && previousRunId && previousLeaseToken) {
      await abandonMinigameRankedRun(previousRunId, previousLeaseToken).catch(() => {});
    }
    releaseRankedOwnership({ forgetLease: ownedPreviousRun });
    if (gameState.value?.gameId) {
      const key = snapshotKey(gameState.value.gameId, difficulty.value);
      if (!rankedPersistenceBlocked) {
        persistState((current) => {
          const activeGameSnapshots = { ...(current.activeGameSnapshots || {}) };
          delete activeGameSnapshots[key];
          return {
            ...current,
            activeGameSnapshots,
          };
        });
      }
    }
    const gameId = String(gameState.value?.gameId || '');
    if (!gameId) return;
    const runtime = await createRankedContext(gameId, difficulty.value);
    await runLocalAction((localController) => localController.startGame(gameId, null, runtime));
  };

  const requestInfo = () => {
    runLocalAction((localController) => localController.requestInfo());
  };

  const triggerCustomAction = ({ key, phase }) => {
    if (!key) return;
    if (Date.now() < inputLockedUntil.value) return;
    runLocalAction((localController) => localController.triggerCustomAction({ key, phase }));
  };

  const usePowerup = (mode) => {
    if (Date.now() < inputLockedUntil.value) return;
    runLocalAction((localController) => localController.usePowerup(mode));
  };

  const cancelInteraction = () => {
    if (Date.now() < inputLockedUntil.value) return;
    runLocalAction((localController) => localController.cancelInteraction());
  };

  const handleBoardCellClick = (cell) => {
    if (!gameState.value?.interaction?.active) return;
    if (Date.now() < inputLockedUntil.value) return;
    runLocalAction((localController) => localController.targetAction(cell.index));
  };

  const move = (direction) => {
    if (currentView.value !== 'play') return;
    if (overlay.value.open && overlay.value.type === 'gameOver') return;
    if (gameState.value?.interaction?.active) return;
    if (Date.now() < inputLockedUntil.value) return;
    runLocalAction((localController) => localController.move(direction));
  };

  const handleKeydown = (event) => {
    if (
      !activeRef.value
      || !keyboardInputAllowed(KEYBOARD_OWNERS.PRIMARY)
      || currentView.value !== 'play'
    ) return;
    if (isTextEntryElement(event.target)) return;
    const map = {
      ArrowUp: 'up',
      ArrowDown: 'down',
      ArrowLeft: 'left',
      ArrowRight: 'right',
      w: 'up',
      a: 'left',
      s: 'down',
      d: 'right',
      W: 'up',
      A: 'left',
      S: 'down',
      D: 'right',
    };
    const direction = map[event.key];
    if (!direction) return;
    event.preventDefault();
    move(direction);
  };

  const handleVisibilityChange = () => {
    if (document.visibilityState === 'visible') {
      void heartbeatActiveRankedRun();
    }
  };

  watch(
    () => authUser.value?.id ?? null,
    (userId) => {
      if (activeRecorder && Number(activeRecorder.userId) !== Number(userId)) {
        releaseRankedOwnership({ forgetLease: true });
        activeRecorder = null;
        rankedPersistenceBlocked = true;
        rankedStatus.value = 'unranked';
      }
    }
  );

  onMounted(() => {
    refreshMenu();
    window.addEventListener('keydown', handleKeydown, true);
    document.addEventListener('visibilitychange', handleVisibilityChange);
    timedTickTimer = window.setInterval(async () => {
      if (timedTickInFlight || gameState.value?.gameId !== 'blitzkrieg') return;
      if (gameState.value?.status === 'game_over') return;
      const panel = (gameState.value?.hud?.customPanels || []).find((item) => item?.type === 'countdown');
      if (!panel?.running) return;
      const remaining = Number(panel.remainingMs || 0) - Math.max(0, Date.now() - Number(panel.syncedAt || Date.now()));
      if (remaining > 0) return;
      timedTickInFlight = true;
      try {
        await runLocalAction((localController) => localController.tick());
      } finally {
        timedTickInFlight = false;
      }
    }, 250);
  });

  onUnmounted(() => {
    if (toastTimer) window.clearTimeout(toastTimer);
    if (inputLockTimer) window.clearTimeout(inputLockTimer);
    if (rankedPollTimer) window.clearTimeout(rankedPollTimer);
    stopRankedHeartbeat();
    if (timedTickTimer) window.clearInterval(timedTickTimer);
    window.removeEventListener('keydown', handleKeydown, true);
    document.removeEventListener('visibilitychange', handleVisibilityChange);
    rankedRunLock.release();
    controller?.close();
    controller = null;
  });

  refreshMenu();

  return {
    menuSections,
    difficulty,
    currentView,
    gameState,
    lastMenuFocusGameId,
    toastMessage,
    overlay,
    rankedStatus,
    closeOverlay,
    setDifficulty,
    startGame,
    backToMenu,
    newGame,
    requestInfo,
    triggerCustomAction,
    usePowerup,
    cancelInteraction,
    handleBoardCellClick,
    move,
  };
}
