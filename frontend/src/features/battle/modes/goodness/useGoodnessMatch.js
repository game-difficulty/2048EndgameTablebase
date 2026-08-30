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

export function useGoodnessMatch(roomSession, activeRef, authUserRef) {
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
  const pendingInputs = new Map();
  let controller = null;
  let routeRoundId = '';
  let frameRevision = 0;
  let overlayTimer = null;
  let correctionResult = null;
  let autoTimer = null;
  let localSequence = 0;

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
    if (result.state.mode === 'input') {
      roomSession.sendModeAction('correction_complete', {
        round_id: room.value.round.round_id,
        sequence: localSequence,
        route_index: result.state.index,
      });
    }
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
    if (!controller || !room.value?.results) {
      opponentBoards.value = {};
      return;
    }
    const frames = {};
    for (const result of room.value.results) {
      if (
        !spectatorMode.value
        && !ownFinished.value
        && Number(result.user_id) !== Number(authUserRef.value?.id)
      ) continue;
      const copy = createBattleController({
        route: controller.route,
        certaintyStep: controller.certaintyStep,
        useVariant: controller.useVariant,
      });
      const state = copy.seek(Number(result.route_index || 0), {
        goodnessOfFit: Number(result.goodness_of_fit ?? 1),
      });
      frames[result.user_id] = createSnapshotBoardFrame(
        `opponent-${result.user_id}-${result.route_index}`,
        state.board,
      );
    }
    opponentBoards.value = frames;
  };

  const syncControllerToServer = () => {
    if (!controller || !ownResult.value || spectatorMode.value) {
      updateOpponentBoards();
      return;
    }
    if (pendingInputs.size > 0) return;
    localSequence = Number(ownResult.value.last_sequence || 0);
    const serverIndex = Number(ownResult.value.route_index || 0);
    const localState = controller.getState();
    const finishingCertaintyRoute = (
      ownResult.value.status === 'completed' && localState.mode === 'auto'
    );
    if (!finishingCertaintyRoute && serverIndex !== Number(localState.index)) {
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
    controller = createBattleController({
      route: payload.buffer,
      certaintyStep: payload.certaintyStep >= 0 ? payload.certaintyStep : null,
      useVariant: useVariant.value,
    });
    routeRoundId = roundId;
    const result = ownResult.value;
    const initialIndex = spectatorMode.value ? 0 : Number(result?.route_index || 0);
    const state = controller.seek(initialIndex, {
      goodnessOfFit: Number(result?.goodness_of_fit ?? 1),
    });
    setFrameSnapshot(state, 'route');
    updateOpponentBoards();
  };

  const onRoomApplied = async (nextRoom) => {
    if (!nextRoom) {
      clearPlaybackTimers();
      controller = null;
      routeRoundId = '';
      pendingInputs.clear();
      controllerState.value = null;
      opponentBoards.value = {};
      return;
    }
    await loadRoute();
    syncControllerToServer();
  };

  const handleMessage = async (message) => {
    if (
      message?.action === 'BATTLE_ACTION_ACCEPTED'
      || message?.action === 'BATTLE_CHOICE_ACCEPTED'
    ) {
      const accepted = message?.data || {};
      pendingInputs.delete(String(accepted.request_id || ''));
      const confirmedResult = room.value?.results?.find(
        (item) => Number(item.user_id) === Number(authUserRef.value?.id),
      );
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
      correctionResult = result;
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
    if (!activeRef.value || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.target?.matches?.('input, textarea, select, [contenteditable="true"]')) return;
    if (event.key === 'Enter' && continueCorrection()) {
      event.preventDefault();
      return;
    }
    const direction = KEY_DIRECTIONS[event.key];
    if (!direction) return;
    if (submitMove(direction)) event.preventDefault();
  };
  const dispose = () => clearPlaybackTimers();

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
      quotaRules.value?.operation_costs?.battle_route_generation ?? 5,
    ),
  }));
  const matchProps = computed(() => ({
    boardFrame: boardFrame.value,
    opponentBoards: opponentBoards.value,
    wrongOverlay: wrongOverlay.value,
    spectator: spectatorMode.value,
    ownFinished: ownFinished.value,
    isVariant: useVariant.value,
  }));
  const matchListeners = Object.freeze({ move: submitMove });
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
    wrongOverlay,
    hallProps,
    matchProps,
    matchListeners,
    createRoom,
    createPracticeJump,
    submitMove,
  };
}
