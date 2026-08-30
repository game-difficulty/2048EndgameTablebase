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
  const wrongOverlay = ref(null);
  const pendingRequest = ref('');
  let currentBoardHex = '';
  let frameRevision = 0;
  let correctionResponse = null;
  let correctionTimer = null;

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
  const applyResolvedStep = (response, key = 'free-step') => {
    const normalized = normalizeHex(response?.board_hex);
    if (!normalized) return;
    const transition = responseTransition(response);
    currentBoardHex = normalized;
    frameRevision += 1;
    boardFrame.value = transition
      ? createTransitionBoardFrame(`${key}-${frameRevision}`, transition, transition.toBoard)
      : createSnapshotBoardFrame(`${key}-${frameRevision}`, boardFromHex(normalized));
    acknowledge(response);
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

  const updateOpponentBoards = () => {
    const frames = {};
    for (const result of room.value?.results || []) {
      const hex = result.mode_data?.board_hex;
      if (!hex) continue;
      if (
        !spectatorMode.value
        && !ownFinished.value
        && Number(result.user_id) !== Number(authUserRef.value?.id)
      ) continue;
      frames[result.user_id] = createSnapshotBoardFrame(
        `free-opponent-${result.user_id}-${result.route_index}`,
        boardFromHex(hex),
      );
    }
    opponentBoards.value = frames;
  };
  const onRoomApplied = async (nextRoom) => {
    if (!nextRoom) {
      clearCorrection();
      pendingRequest.value = '';
      currentBoardHex = '';
      opponentBoards.value = {};
      return;
    }
    const own = nextRoom.results?.find(
      (item) => Number(item.user_id) === Number(authUserRef.value?.id),
    );
    const stateStatus = String(own?.mode_data?.state_status || '');
    if (['input', 'finished'].includes(stateStatus)) {
      pendingRequest.value = '';
    }
    const lastStep = own?.mode_data?.last_step;
    const recoverCorrection = (
      stateStatus === 'awaiting_ack'
      && Boolean(lastStep?.corrected)
      && !correctionResponse
    );
    if (recoverCorrection) {
      pendingRequest.value = pendingRequest.value || 'recovered-correction';
      showCorrection({
        ...lastStep,
        round_id: nextRoom.round?.round_id,
        sequence: Number(lastStep.sequence || own?.last_sequence || 0),
        awaiting_ack: true,
      });
    } else if (!correctionResponse) {
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
      sequence: Number(ownResult.value?.last_sequence || 0) + 1,
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
  const dispose = () => clearCorrection();

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
    costLabelKey: 'battle.form.freeModeBudget',
    refundPolicyKey: 'battle.form.freeModeRefundPolicy',
    createLabelKey: 'battle.actions.createFree',
  }));
  const matchProps = computed(() => ({
    boardFrame: boardFrame.value,
    opponentBoards: opponentBoards.value,
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
