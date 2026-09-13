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
import { createBattleController } from './engine/battleController.js';
import { battleClient, battleRequestId } from '../../services/battleClient.js';
import { correctionOverlayForResult } from '../../core/battleCorrection.js';
import { isBattlePlaybackStopped } from '../../core/battlePlaybackState.js';
import { createObserverPlayback } from '../../core/observerPlayback.js';
import { createRoutePresentation } from './engine/routePresentation.js';

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
  W: 'up',
  S: 'down',
});
const CORRECTION_TIMEOUT_MS = 15_000;

function transitionFrame(transition) {
  if (!transition) return null;
  return {
    fromBoard: transition.board,
    toBoard: transition.nextBoard,
    metadata: transition.metadata,
  };
}

export function useGoodnessMatch(
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
  const wrongOverlay = ref(null);
  const boardFrame = ref(
    createSnapshotBoardFrame('battle-empty', new Array(16).fill(0)),
  );
  const controllerState = ref(null);
  const opponentBoards = ref({});
  const opponentOverlays = ref({});
  const pendingInputs = new Map();
  const pendingCorrections = new Map();
  let controller = null;
  let routeRoundId = '';
  let frameRevision = 0;
  let overlayTimer = null;
  let correctionResult = null;
  let autoTimer = null;
  let localSequence = 0;
  let routePresentation = null;
  const observer = createObserverPlayback({
    resolve: (result, time) => routePresentation?.(result, time),
    canSee: (result) => spectatorMode.value || ownFinished.value || roomSession.isOwnActor(result),
    publish: (frames, overlays) => {
      opponentBoards.value = frames;
      opponentOverlays.value = overlays;
    },
  });

  const catalogGroups = computed(() => groupTablebasesByPattern(catalog.value));
  const multiplierForPattern = (fullPattern) => {
    const base = String(fullPattern || '').split('_', 1)[0];
    const rules = quotaRules.value?.table_groups || [];
    for (const group of rules) {
      if ((group.patterns || []).some((prefix) => base.startsWith(String(prefix)))) {
        return Number(group.multiplier || 1);
      }
    }
    return Number(quotaRules.value?.default_multiplier || 1);
  };
  const inputEnabled = computed(() => (
    Boolean(activeRef.value)
    && String(room.value?.mode_key || 'goodness') === 'goodness'
    && matchActive.value
    && !spectatorMode.value
    && ownResult.value?.status === 'playing'
    && controllerState.value?.mode === 'input'
    && !wrongOverlay.value
  ));
  const useVariant = computed(() => isVariantPattern(room.value?.pattern, {
    variant: ['2x4', '3x3', '3x4', '3x4free9', '3x3free8'],
  }));

  const setFrameSnapshot = (state, key = 'sync') => {
    if (!state) return;
    frameRevision += 1;
    boardFrame.value = createSnapshotBoardFrame(`${key}-${frameRevision}`, state.board);
    controllerState.value = state;
  };
  const animateTransition = (result, key = 'step') => {
    frameRevision += 1;
    boardFrame.value = createTransitionBoardFrame(
      `${key}-${frameRevision}`,
      transitionFrame(result?.transition),
      result?.state?.board,
    );
    controllerState.value = result?.state || controllerState.value;
  };
  const clearPlaybackTimers = () => {
    if (overlayTimer != null) window.clearTimeout(overlayTimer);
    if (autoTimer != null) window.clearTimeout(autoTimer);
    overlayTimer = null;
    autoTimer = null;
    correctionResult = null;
    wrongOverlay.value = null;
  };
  const continueCorrection = () => {
    if (!wrongOverlay.value || !correctionResult) return false;
    if (overlayTimer != null) window.clearTimeout(overlayTimer);
    overlayTimer = null;
    const result = correctionResult;
    correctionResult = null;
    wrongOverlay.value = null;
    const payload = {
      round_id: result.roundId,
      sequence: result.sequence,
      route_index: result.state.index,
    };
    const requestId = battleRequestId('correction');
    pendingCorrections.set(requestId, payload);
    roomSession.sendModeAction('correction_complete', payload, { requestId });
    animateTransition(result, 'wrong-correction');
    if (result.state.mode === 'auto') runAutoPlayback();
    return true;
  };
  const runAutoPlayback = () => {
    if (!controller || controller.getState().mode !== 'auto') return;
    autoTimer = window.setTimeout(() => {
      const result = controller.autoStep();
      if (result.accepted) animateTransition(result, 'auto');
      if (controller?.getState().mode === 'auto') runAutoPlayback();
    }, 150);
  };

  const updateOpponentBoards = () => {
    observer.update(controller ? room.value : null);
  };

  const syncControllerToServer = () => {
    if (!controller || !ownResult.value || spectatorMode.value) {
      updateOpponentBoards();
      return;
    }
    if (pendingInputs.size > 0) return;
    if (correctionResult) { updateOpponentBoards(); return; }
    localSequence = Number(ownResult.value.last_sequence || 0);
    const serverIndex = Number(ownResult.value.route_index || 0);
    const localState = controller.getState();
    const finishingCertaintyRoute = (
      ownResult.value.status === 'completed' && localState.mode === 'auto'
    );
    if (isBattlePlaybackStopped(ownResult.value) || (!finishingCertaintyRoute && serverIndex !== Number(localState.index))) {
      const state = controller.seek(serverIndex, {
        goodnessOfFit: Number(ownResult.value.goodness_of_fit ?? 1),
      });
      setFrameSnapshot(state, 'server');
    }
    updateOpponentBoards();
  };

  const loadRoute = async () => {
    const roundId = String(room.value?.round?.round_id || '');
    if (!roundId || !room.value?.route || routeRoundId === roundId) {
      syncControllerToServer();
      return;
    }
    const payload = await battleClient.route(room.value.room_code, roundId);
    if (room.value?.round?.round_id !== roundId) return;
    if (routeRoundId === roundId) { syncControllerToServer(); return; }
    clearPlaybackTimers();
    observer.clear();
    pendingInputs.clear();
    pendingCorrections.clear();
    controller = createBattleController({
      route: payload.buffer,
      certaintyStep: payload.certaintyStep >= 0 ? payload.certaintyStep : null,
      useVariant: useVariant.value,
    });
    routePresentation = createRoutePresentation(controller);
    routeRoundId = roundId;
    const result = ownResult.value;
    localSequence = Number(result?.last_sequence || 0);
    const initialIndex = spectatorMode.value ? 0 : Number(result?.route_index || 0);
    const state = controller.seek(initialIndex, {
      goodnessOfFit: Number(result?.goodness_of_fit ?? 1),
    });
    setFrameSnapshot(state, 'route');
    if (!spectatorMode.value && result && !isBattlePlaybackStopped(result)) {
      const correction = correctionOverlayForResult(result);
      if (correction) {
        setFrameSnapshot(controller.seek(correction.previousRouteIndex), 'restore-correction');
        const restored = controller.input(correction.selectedDirection);
        if (restored.accepted) {
          controller.goodnessOfFit = Number(result.goodness_of_fit ?? 1);
          restored.state = controller.getState();
          correctionResult = { ...restored, roundId, sequence: Number(result.last_sequence) };
          wrongOverlay.value = correction;
          overlayTimer = window.setTimeout(continueCorrection,
            Math.max(0, correction.visibleUntil - Date.now()));
        }
      } else if (result.status === 'completed' && result.mode_data?.auto_playback) {
        const view = routePresentation(result, Date.now());
        setFrameSnapshot(controller.seek(view.index, { goodnessOfFit: result.goodness_of_fit }), 'restore-auto');
        runAutoPlayback();
      }
    }
    updateOpponentBoards();
  };

  const onRoomApplied = async (nextRoom) => {
    if (!nextRoom) {
      clearPlaybackTimers();
      observer.clear();
      controller = null;
      routePresentation = null;
      routeRoundId = '';
      pendingInputs.clear();
      pendingCorrections.clear();
      controllerState.value = null;
      opponentBoards.value = {};
      opponentOverlays.value = {};
      return;
    }
    if (isBattlePlaybackStopped(ownResult.value)) {
      clearPlaybackTimers();
      pendingInputs.clear();
      pendingCorrections.clear();
    }
    await loadRoute();
    for (const [requestId, payload] of pendingCorrections) {
      if (payload.round_id !== routeRoundId || Number(ownResult.value?.last_sequence) !== payload.sequence
          || !ownResult.value?.mode_data?.correction) {
        pendingCorrections.delete(requestId);
      } else {
        roomSession.sendModeAction('correction_complete', payload, { requestId });
      }
    }
    syncControllerToServer();
  };

  const handleMessage = async (message) => {
    if (
      message?.action === 'BATTLE_ACTION_ACCEPTED'
      || message?.action === 'BATTLE_CHOICE_ACCEPTED'
    ) {
      const accepted = message?.data || {};
      if (accepted.round_id && accepted.round_id !== routeRoundId) return true;
      if (isBattlePlaybackStopped(ownResult.value)) return true;
      if (accepted.kind === 'correction_complete') {
        pendingCorrections.delete(String(accepted.request_id || ''));
        return true;
      }
      pendingInputs.delete(String(accepted.request_id || ''));
      const confirmedResult = room.value?.results?.find(roomSession.isOwnActor);
      if (confirmedResult) {
        confirmedResult.last_sequence = Number(
          accepted.sequence ?? confirmedResult.last_sequence,
        );
        confirmedResult.route_index = Number(
          accepted.route_index ?? confirmedResult.route_index,
        );
        confirmedResult.goodness_of_fit = Number(
          accepted.goodness_of_fit ?? confirmedResult.goodness_of_fit,
        );
        if (accepted.timeout_at) confirmedResult.timeout_at = accepted.timeout_at;
        if (accepted.complete) confirmedResult.status = 'completed';
      }
      syncControllerToServer();
      return true;
    }
    if (
      message?.action === 'BATTLE_ACTION_CONFLICT'
      || message?.action === 'BATTLE_PROGRESS_CONFLICT'
    ) {
      if (pendingCorrections.delete(String(message?.data?.request_id || ''))) {
        await roomSession.refreshCurrent();
        return true;
      }
      pendingInputs.clear();
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
    mode_key: 'goodness',
  });

  const submitMove = (direction) => {
    if (!inputEnabled.value || !controller) return false;
    const result = controller.input(direction);
    if (!result.accepted) return false;
    const requestId = battleRequestId('move');
    pendingInputs.set(requestId, result.stepIndex);
    localSequence += 1;
    roomSession.sendModeAction('move', {
      round_id: room.value.round.round_id,
      sequence: localSequence,
      route_index: result.stepIndex,
      direction: result.selectedDirection,
    }, { requestId });
    if (result.wrong) {
      if (overlayTimer != null) window.clearTimeout(overlayTimer);
      correctionResult = { ...result, roundId: routeRoundId, sequence: localSequence };
      wrongOverlay.value = {
        selectedDirection: result.selectedDirection,
        standardDirection: result.standardDirection,
        drop: result.scoring.goodnessDrop,
      };
      overlayTimer = window.setTimeout(() => {
        continueCorrection();
      }, CORRECTION_TIMEOUT_MS);
    } else {
      animateTransition(result, 'move');
      if (result.state.mode === 'auto') runAutoPlayback();
    }
    return true;
  };

  const handleKeydown = (event) => {
    if (
      !hotkeysEnabledRef.value
      || String(room.value?.mode_key || 'goodness') !== 'goodness'
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
    if (!direction) return;
    if (submitMove(direction)) event.preventDefault();
  };
  const dispose = () => {
    clearPlaybackTimers();
    observer.clear();
  };

  roomSession.registerModeAdapter({
    key: 'goodness',
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
    routeBaseCost: Number(
      quotaRules.value?.operation_costs?.battle_route_generation ?? 100,
    ) * Number(quotaRules.value?.global_multiplier ?? 1),
  }));
  const matchProps = computed(() => ({
    boardFrame: boardFrame.value,
    opponentBoards: opponentBoards.value,
    opponentOverlays: opponentOverlays.value,
    wrongOverlay: wrongOverlay.value,
    spectator: spectatorMode.value,
    ownFinished: ownFinished.value,
    isVariant: useVariant.value,
  }));
  const matchListeners = Object.freeze({
    move: submitMove,
    'continue-correction': continueCorrection,
  });
  const createPracticeJump = () => {
    const state = controllerState.value;
    if (!state?.boardHex || !room.value?.full_pattern) return null;
    return {
      source: 'battle',
      fullPattern: String(room.value.full_pattern),
      hex: String(state.boardHex),
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
    boardFrame,
    controllerState,
    opponentBoards,
    opponentOverlays,
    wrongOverlay,
    hallProps,
    matchProps,
    matchListeners,
    createRoom,
    createPracticeJump,
    submitMove,
  };
}
