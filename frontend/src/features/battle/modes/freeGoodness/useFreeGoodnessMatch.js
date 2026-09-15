import {
  computed,
  onMounted,
  onUnmounted,
  ref,
} from 'vue';

import {
  createSnapshotBoardFrame,
  createTransitionBoardFrame,
} from '../../../../components/boardFrame.js';
import { getBackendUrl } from '../../../../services/runtime/backendUrl.js';
import {
  fetchTablebaseCatalog,
  groupTablebasesByPattern,
} from '../../../../services/tablebases/catalogClient.js';
import { isVariantPattern } from '../../../../utils/patternCategories.js';
import {
  boardHex,
  buildOptimisticMoveOnlyTransition,
  decodeBoard,
  encodeBoard,
} from '../../../replay/engine/replayTransition.js';
import { battleRequestId } from '../../services/battleClient.js';
import { correctionOverlayForResult } from '../../core/battleCorrection.js';
import { isBattlePlaybackStopped } from '../../core/battlePlaybackState.js';
import { createObserverPlayback } from '../../core/observerPlayback.js';

const KEY_DIRECTIONS = Object.freeze({
  ArrowLeft: 'left',
  ArrowRight: 'right',
  ArrowUp: 'up',
  ArrowDown: 'down',
  a: 'left',
  d: 'right',
  w: 'up',
  s: 'down',
  A: 'left',
  D: 'right',
  h: 'left',
  j: 'down',
  k: 'up',
  l: 'right',
  H: 'left',
  J: 'down',
  K: 'up',
  L: 'right',
  W: 'up',
  S: 'down',
});
const CORRECTION_TIMEOUT_MS = 15_000;
const AUTO_PLAYBACK_STEP_MS = 150;

const normalizeHex = (value) => {
  const text = String(value || '').trim().replace(/^0x/iu, '').toLowerCase();
  return /^[0-9a-f]{1,16}$/u.test(text) ? text.padStart(16, '0') : '';
};

const boardFromHex = (value) => {
  const normalized = normalizeHex(value);
  return normalized ? decodeBoard(BigInt(`0x${normalized}`)) : new Array(16).fill(0);
};

export function useFreeGoodnessMatch(
  roomSession,
  activeRef,
  authUserRef,
  hotkeysEnabledRef = activeRef,
) {
  const {
    room,
    error,
    ownResult,
    matchActive,
    spectatorMode,
    ownFinished,
  } = roomSession;
  const catalog = ref([]);
  const quotaRules = ref(null);
  const boardFrame = ref(createSnapshotBoardFrame('free-battle-empty', new Array(16).fill(0)));
  const opponentBoards = ref({});
  const opponentOverlays = ref({});
  const wrongOverlay = ref(null);
  const pendingRequest = ref('');
  let currentBoardHex = '';
  let localRoundId = '';
  let localSequence = 0;
  let frameRevision = 0;
  let correctionResponse = null;
  let correctionTimer = null;
  let autoPlaybackTimer = null;
  let autoPlaybackActive = false;
  let autoPlaybackFinalHex = '';
  let autoPlaybackSteps = [];
  const seenAutoPlaybackKeys = new Set();

  const catalogGroups = computed(() => groupTablebasesByPattern(catalog.value));
  const useVariant = computed(() => isVariantPattern(room.value?.pattern, {
    variant: ['2x4', '3x3', '3x4', '3x4free9', '3x3free8'],
  }));
  const multiplierForPattern = (fullPattern) => {
    const base = String(fullPattern || '').split('_', 1)[0];
    for (const group of quotaRules.value?.table_groups || []) {
      if ((group.patterns || []).some((prefix) => base.startsWith(String(prefix)))) {
        return Number(group.multiplier || 1);
      }
    }
    return Number(quotaRules.value?.default_multiplier || 1);
  };
  const calculateCost = ({ full_pattern: fullPattern, target, max_players: maxPlayers }) => (
    Number(maxPlayers || 2)
    * (Math.trunc(Number(target) || 0) / 2 + 10)
    * multiplierForPattern(fullPattern)
    * Number(quotaRules.value?.global_multiplier ?? 1)
  );
  const inputEnabled = computed(() => (
    Boolean(activeRef.value)
    && room.value?.mode_key === 'free_goodness'
    && matchActive.value
    && !spectatorMode.value
    && ownResult.value?.status === 'playing'
    && ownResult.value?.mode_data?.state_status === 'input'
    && !pendingRequest.value
    && !wrongOverlay.value
  ));

  const setSnapshot = (hex, key = 'sync') => {
    const normalized = normalizeHex(hex);
    if (!normalized || normalized === currentBoardHex) return;
    currentBoardHex = normalized;
    frameRevision += 1;
    boardFrame.value = createSnapshotBoardFrame(
      `${key}-${frameRevision}`,
      boardFromHex(normalized),
    );
  };
  const responseTransition = (response) => {
    const fromBoard = boardFromHex(response.previous_board_hex || currentBoardHex);
    const toBoard = boardFromHex(response.board_hex);
    const moved = buildOptimisticMoveOnlyTransition(
      fromBoard,
      response.executed_direction,
      useVariant.value,
    );
    if (!moved) return null;
    const metadata = { ...moved.metadata };
    if (Number(response.spawn_index) >= 0 && [2, 4].includes(Number(response.spawn_value))) {
      metadata.appear_tile = {
        index: Number(response.spawn_index),
        value: Number(response.spawn_value),
      };
    }
    return { fromBoard, toBoard, metadata };
  };
  const acknowledge = (response) => {
    if (!response?.awaiting_ack) {
      pendingRequest.value = '';
      return;
    }
    pendingRequest.value = roomSession.sendModeAction('step_ready_ack', {
      round_id: response.round_id,
      sequence: response.sequence,
    });
  };
  const renderResolvedStep = (response, key = 'free-step') => {
    const normalized = normalizeHex(response?.board_hex);
    if (!normalized) return false;
    const transition = responseTransition(response);
    currentBoardHex = normalized;
    frameRevision += 1;
    boardFrame.value = transition
      ? createTransitionBoardFrame(`${key}-${frameRevision}`, transition, transition.toBoard)
      : createSnapshotBoardFrame(`${key}-${frameRevision}`, boardFromHex(normalized));
    return true;
  };
  const clearAutoPlayback = () => {
    if (autoPlaybackTimer != null) window.clearTimeout(autoPlaybackTimer);
    autoPlaybackTimer = null;
    autoPlaybackActive = false;
    autoPlaybackFinalHex = '';
    autoPlaybackSteps = [];
  };
  const playNextAutoStep = () => {
    if (!autoPlaybackActive) return;
    const step = autoPlaybackSteps.shift();
    if (!step) {
      const finalHex = autoPlaybackFinalHex;
      clearAutoPlayback();
      if (finalHex) setSnapshot(finalHex, 'free-certainty-final');
      pendingRequest.value = '';
      return;
    }
    renderResolvedStep(step, 'free-certainty-auto');
    autoPlaybackTimer = window.setTimeout(playNextAutoStep, AUTO_PLAYBACK_STEP_MS);
  };
  const startAutoPlayback = (response, { syncStart = false } = {}) => {
    const steps = Array.isArray(response?.auto_steps)
      ? response.auto_steps.filter((step) => normalizeHex(step?.board_hex))
      : [];
    if (!steps.length) return false;
    const key = String(response?.auto_playback_key || '');
    if (key && seenAutoPlaybackKeys.has(key)) return false;
    if (key) seenAutoPlaybackKeys.add(key);
    clearAutoPlayback();
    autoPlaybackActive = true;
    autoPlaybackSteps = [...steps];
    autoPlaybackFinalHex = normalizeHex(
      response?.auto_final_board_hex || steps[steps.length - 1]?.board_hex,
    );
    pendingRequest.value = pendingRequest.value || 'certainty-auto';
    if (syncStart) {
      setSnapshot(
        response?.auto_start_board_hex || steps[0]?.previous_board_hex,
        'free-certainty-start',
      );
    }
    autoPlaybackTimer = window.setTimeout(playNextAutoStep, AUTO_PLAYBACK_STEP_MS);
    return true;
  };
  const applyResolvedStep = (response, key = 'free-step') => {
    if (!renderResolvedStep(response, key)) return;
    if (!startAutoPlayback(response)) acknowledge(response);
  };
  const clearCorrection = () => {
    if (correctionTimer != null) window.clearTimeout(correctionTimer);
    correctionTimer = null;
    correctionResponse = null;
    wrongOverlay.value = null;
  };
  const continueCorrection = () => {
    if (!correctionResponse) return false;
    const response = correctionResponse;
    clearCorrection();
    applyResolvedStep(response, 'free-correction');
    return true;
  };
  const showCorrection = (response) => {
    correctionResponse = response;
    wrongOverlay.value = {
      selectedDirection: response.selected_direction,
      standardDirection: response.executed_direction,
      drop: Math.max(0, 1 - Number(response.step_goodness || 0)),
    };
    if (correctionTimer != null) window.clearTimeout(correctionTimer);
    correctionTimer = window.setTimeout(continueCorrection, CORRECTION_TIMEOUT_MS);
  };

  const observer = createObserverPlayback({
    canSee: (result) => spectatorMode.value || ownFinished.value || roomSession.isOwnActor(result),
    resolve: (result, time) => {
      const correction = correctionOverlayForResult(result, time);
      const hex = correction?.previousBoardHex || result.mode_data?.board_hex;
      return hex ? { board: boardFromHex(hex), index: Number(result.route_index || 0),
        overlay: correction, nextAt: correction?.visibleUntil } : null;
    },
    publish: (frames, overlays) => {
      opponentBoards.value = frames;
      opponentOverlays.value = overlays;
    },
  });
  const updateOpponentBoards = () => {
    observer.update(room.value);
  };
  const onRoomApplied = async (nextRoom) => {
    if (!nextRoom) {
      observer.clear();
      clearCorrection();
      clearAutoPlayback();
      seenAutoPlaybackKeys.clear();
      pendingRequest.value = '';
      currentBoardHex = '';
      localRoundId = '';
      localSequence = 0;
      opponentBoards.value = {};
      opponentOverlays.value = {};
      return;
    }
    const own = nextRoom.results?.find(roomSession.isOwnActor);
    const nextRoundId = String(nextRoom.round?.round_id || '');
    if (nextRoundId !== localRoundId) {
      localRoundId = nextRoundId;
      localSequence = Number(own?.last_sequence || 0);
    } else {
      localSequence = Math.max(localSequence, Number(own?.last_sequence || 0));
    }
    const stateStatus = String(own?.mode_data?.state_status || '');
    if (isBattlePlaybackStopped(own)) {
      clearCorrection();
      clearAutoPlayback();
      pendingRequest.value = '';
      setSnapshot(own.mode_data?.board_hex, 'free-stopped');
      updateOpponentBoards();
      return;
    }
    if (['input', 'finished'].includes(stateStatus) && !autoPlaybackActive && !correctionResponse) {
      pendingRequest.value = '';
    }
    const lastStep = own?.mode_data?.last_step;
    const recoverCorrection = (
      stateStatus === 'awaiting_ack'
      && Boolean(lastStep?.corrected)
      && !correctionResponse
    );
    const recoverAutoPlayback = (
      !correctionResponse
      && !autoPlaybackActive
      && Array.isArray(own?.mode_data?.auto_steps)
      && own.mode_data.auto_steps.length > 0
      && !seenAutoPlaybackKeys.has(String(own.mode_data.auto_playback_key || ''))
    );
    if (recoverCorrection) {
      pendingRequest.value = pendingRequest.value || 'recovered-correction';
      showCorrection({
        ...lastStep,
        round_id: nextRoom.round?.round_id,
        sequence: Number(lastStep.sequence || own?.last_sequence || 0),
        awaiting_ack: true,
      });
    } else if (recoverAutoPlayback) {
      startAutoPlayback(own.mode_data, { syncStart: true });
    } else if (!correctionResponse && !autoPlaybackActive) {
      setSnapshot(own?.mode_data?.board_hex, 'free-room');
    }
    updateOpponentBoards();
    if (stateStatus === 'awaiting_ack' && !correctionResponse) {
      roomSession.sendModeAction('step_ready_ack', {
        round_id: nextRoom.round?.round_id,
        sequence: Number(own.last_sequence || 0),
      });
    }
  };
  const handleMessage = async (message) => {
    if (message?.action === 'BATTLE_ACTION_ACCEPTED') {
      const response = message.data || {};
      if (response.round_id && response.round_id !== localRoundId) return true;
      if (isBattlePlaybackStopped(ownResult.value)) return true;
      localSequence = Math.max(localSequence, Number(response.sequence || 0));
      if (response.board_hex) {
        if (response.corrected) showCorrection(response);
        else applyResolvedStep(response);
      } else if (response.timeout_at || response.complete) {
        pendingRequest.value = '';
      }
      return true;
    }
    if (message?.action === 'BATTLE_ACTION_CONFLICT') {
      pendingRequest.value = '';
      error.value = message?.data?.code || 'PROGRESS_CONFLICT';
      await roomSession.refreshCurrent();
      return true;
    }
    return false;
  };
  const bootstrap = async () => {
    const [tables, rules] = await Promise.all([
      fetchTablebaseCatalog(),
      fetch(getBackendUrl('/api/quota/rules')).then((response) => response.json()),
    ]);
    catalog.value = tables;
    quotaRules.value = rules;
  };
  const createRoom = (payload) => roomSession.createRoom({
    ...payload,
    max_steps: null,
    mode_key: 'free_goodness',
  });
  const submitMove = (direction) => {
    if (!inputEnabled.value) return false;
    const requestId = battleRequestId('free-move');
    pendingRequest.value = requestId;
    roomSession.sendModeAction('move', {
      round_id: room.value.round?.round_id,
      sequence: Math.max(
        localSequence,
        Number(ownResult.value?.last_sequence || 0),
      ) + 1,
      direction,
    }, { requestId });
    return true;
  };
  const handleKeydown = (event) => {
    if (
      !hotkeysEnabledRef.value
      || room.value?.mode_key !== 'free_goodness'
      || event.ctrlKey
      || event.metaKey
      || event.altKey
    ) return;
    if (event.target?.matches?.('input, textarea, select, [contenteditable="true"]')) return;
    if (event.key === 'Enter' && continueCorrection()) {
      event.preventDefault();
      return;
    }
    const direction = KEY_DIRECTIONS[event.key];
    if (direction && submitMove(direction)) event.preventDefault();
  };
  const dispose = () => {
    observer.clear();
    clearCorrection();
    clearAutoPlayback();
  };

  roomSession.registerModeAdapter({
    key: 'free_goodness',
    bootstrap,
    onRoomApplied,
    handleMessage,
    dispose,
  });
  onMounted(() => window.addEventListener('keydown', handleKeydown, { capture: true }));
  onUnmounted(() => window.removeEventListener('keydown', handleKeydown, { capture: true }));

  const hallProps = computed(() => ({
    tables: catalog.value,
    multiplierForPattern,
    calculateCost,
    showMaxSteps: false,
    showRankingMinSteps: true,
    costLabelKey: 'battle.form.freeModeBudget',
    refundPolicyKey: 'battle.form.freeModeRefundPolicy',
    createLabelKey: 'battle.actions.createFree',
  }));
  const matchProps = computed(() => ({
    boardFrame: boardFrame.value,
    opponentBoards: opponentBoards.value,
    opponentOverlays: opponentOverlays.value,
    wrongOverlay: wrongOverlay.value,
    resolving: Boolean(pendingRequest.value),
    spectator: spectatorMode.value,
    ownFinished: ownFinished.value,
    isVariant: useVariant.value,
  }));
  const matchListeners = Object.freeze({
    move: submitMove,
    'continue-correction': continueCorrection,
  });
  const createPracticeJump = () => {
    if (!currentBoardHex || !room.value?.full_pattern) return null;
    return {
      source: 'battle',
      fullPattern: String(room.value.full_pattern),
      hex: currentBoardHex,
      preferDock: true,
      claimKeyboard: false,
      queryPolicy: 'same-table-only',
      context: {
        kind: 'battle',
        roomId: String(room.value.room_id || ''),
        roundId: String(room.value.round?.round_id || ''),
        fullPattern: String(room.value.full_pattern),
      },
    };
  };

  return {
    catalog,
    catalogGroups,
    multiplierForPattern,
    hallProps,
    matchProps,
    matchListeners,
    createRoom,
    createPracticeJump,
    submitMove,
  };
}
