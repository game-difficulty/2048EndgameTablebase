<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue';
import { language, t } from './i18n.js';
import LanguageSwitch from './LanguageSwitch.vue';
import { vTouchClick } from './touchClick.js';
import { api, connectRoom, projectSocket, projectAuthentication } from './api';
import { ProjectStreamSender } from './projects/projectStream.js';
import { ServerClock } from './serverClock.js';
import { userFacingError } from './errorMessages.js';
import ProjectPlayground from './projects/ProjectPlayground.vue';
import PlayerAvatar from './PlayerAvatar.vue';
import EventCenter from './EventCenter.vue';
import MatchSettlement from '../../shared/MatchSettlement.vue';
import RoomRulesPicker from './RoomRulesPicker.vue';
import DraftWorkflow from '../../shared/DraftWorkflow.vue';
import { lineupValid, lineupPolicyDescription } from './roomRules.js';
import RoomSchedulePicker from './RoomSchedulePicker.vue';
import RoomMemberManagement from './RoomMemberManagement.vue';
import { scheduleTime, roomMatchTitle, roomMatchScore } from './scheduleDisplay.js';
import TournamentBoard from './projects/TournamentBoard.vue';
import CargoBoard from './projects/CargoBoard.vue';
import ObservedProjectBoard from '../../shared/ObservedProjectBoard.vue';
import PolyominoBoard from './projects/PolyominoBoard.vue';
import { projectIconUrl } from '../../shared/projectIcons.js';
import { projectPerformanceMetric, projectRuleMetrics, projectResultValue as rawProjectResultValue } from '../../shared/projectMetrics.mjs';
import { ownDeadline, phaseSeconds, stageChange } from './stageMotion.js';
import { MatchRuntime } from './projects/matchRuntime.js';
import { receivedProjectView } from '../../shared/projectStateOrder.mjs';
import {
  competitionProjectInput,
  PROJECT_BY_ID,
  PROJECT_BY_ORDER,
  TOURNAMENT_PROJECTS,
} from './projects/catalog.js';

const pathname = ref(window.location.pathname);
const projectRouteMatch = computed(() => pathname.value.match(/^\/projects(?:\/([^/]+))?\/?$/));
const practiceRouteMatch = computed(() => pathname.value.match(/^\/practice(?:\/(20|1[0-9]|[1-9]))?\/?$/));
const isProjectRoute = computed(() => Boolean(projectRouteMatch.value || practiceRouteMatch.value));
const projectRouteId = computed(() => (projectRouteMatch.value?.[1]
  ? decodeURIComponent(projectRouteMatch.value[1])
  : PROJECT_BY_ORDER[Number(practiceRouteMatch.value?.[1])]?.id || ''));

const session = ref(null);
const rooms = ref([]);
const events = ref([]);
const canCreateEvent = ref(false);
const eventSlug = computed(() => pathname.value.match(/^\/events\/([a-z0-9-]+)\/?$/)?.[1] || '');
const eventHasRooms = computed(() => !eventSlug.value || events.value.find(event => event.slug === eventSlug.value)?.capabilities.rooms !== false);
const newRoomEventSlug = ref('');
const newRoomSchedule = ref({});
const newRoomRules = ref(null);
const newRoomRulesValid = ref(false);
const createRoomOpen = ref(false);
const room = ref(null);
const loading = ref(true);
const busy = ref(false);
const error = ref('');
const connection = ref('offline');
const newRoomName = ref('2048 团队赛');
const selectedProjectIds = ref(TOURNAMENT_PROJECTS.map(project => project.id));
const joinCode = ref('');
const selectedPick = ref('');
const selectedBan = ref('');
const selectedBlind = ref('');
const lineupSelections = ref({ A: 1, B: 2, C: 3 });
const issueCategory = ref('network_device');
const issueDetails = ref('');
const issueOpen = ref(false);
const manageUserId = ref('');
const issueResolutionNote = ref('');
const suspendReasonCode = ref('network_device');
const suspendReasonText = ref('');
const overrideYellowScore = ref(0);
const overrideWhiteScore = ref(0);
const overrideWinner = ref('draw');
const overrideReason = ref('');
const forceAdvanceReason = ref('');
const forceFinishWinner = ref('draw');
const forceFinishReason = ref('');
const clockNow = ref(Date.now());
const serverClock = new ServerClock();
let roomRefreshPending = false;
const predictionWait = computed(() => Math.max(0, Math.ceil((Date.parse(match.value?.prediction_window?.minimum_until || '') - clockNow.value) / 1000)) || 0);
const movePending = ref(false);
const projectThinking = ref(false);
const observerPending = ref({ yellow: false, white: false });
const finishPlaybackUntil = ref(0);
const localPacket = ref(null);
const projectSyncState = ref('synced');
let localRuntime = null;
let stateSender = null;
let localSaveTimer = null;
let acknowledgedSequence = 0;
const stageMotion = ref(null);
const revealStep = ref(2);
const mainSiteUrl = String(import.meta.env.VITE_MAIN_SITE_URL || 'https://2048tables.online/');
const competitionHomePath = String(import.meta.env.VITE_COMPETITION_HOME_PATH || '/test');
let disconnectRoom = null;
let ticker = null;
let routeEpoch = 0;
let previousPhaseToken = '';
let previousLineupToken = '';
let previousResultKey = '';
let projectThinkingTimer = null;
let stageMotionTimer = null;
let revealTimers = [];

const statusText = {
  SEATING: '选手落座',
  READY_CHECK: '队长准备',
  DRAW: '先后手抽签',
  DRAFT_STEP: '项目选禁',
  FIRST_PICK_BAN: '先手选择',
  SECOND_PICK_BAN: '后手选择',
  BLIND_PICK: '双方盲选',
  C_DRAW: 'BP 完成',
  LINEUP: '秘密布阵',
  GAME_A_READY: '阵容公布',
  GAME_A_PLAYING: '项目 A 对局',
  GAME_A_RESULT: '项目 A 结果',
  GAME_B_READY: '项目 B 开局检查',
  GAME_B_PLAYING: '项目 B 对局',
  GAME_B_RESULT: '项目 B 结果',
  GAME_C_READY: '项目 C 开局检查',
  GAME_C_PLAYING: '项目 C 对局',
  GAME_C_RESULT: '项目 C 结果',
  FINISHED: '比赛结束',
  CANCELLED: '房间已关闭',
};

for (const key of 'DEFGHIJKLMNO') for (const [phase,label] of Object.entries({READY:'开局检查',PLAYING:'对局',RESULT:'结果'})) statusText[`GAME_${key}_${phase}`] = `项目 ${key} ${label}`;

const currentCode = computed(() => {
  const match = pathname.value.match(/^\/rooms\/([A-Za-z0-9]+)\/?$/);
  return match ? match[1].toUpperCase() : '';
});
const selectedProjects = computed(() => selectedProjectIds.value.map(id => PROJECT_BY_ID[id]).filter(Boolean));
const activeRooms = computed(() => rooms.value.filter(item => item.status !== 'CANCELLED').sort((a,b) => (a.schedule?.starts_at || a.created_at).localeCompare(b.schedule?.starts_at || b.created_at)));
const closedRooms = computed(() => rooms.value.filter(item => item.status === 'CANCELLED'));
const orderedProjectOptions = computed(() => [
  ...selectedProjects.value,
  ...TOURNAMENT_PROJECTS.filter(project => !selectedProjectIds.value.includes(project.id)),
]);

const seats = computed(() => {
  const map = new Map();
  for (const seat of room.value?.seats || []) map.set(`${seat.side}:${seat.position}`, seat);
  return map;
});

const mySeat = computed(() => room.value?.me?.seat || null);
const myTeamReady = computed(() => {
  const side = mySeat.value?.side;
  return side ? Boolean(room.value?.teams?.[side]?.ready) : false;
});
const roomGameKeys = computed(() => room.value?.rules?.game_keys || ['A','B','C']);
const roomPositions = computed(() => Array.from({length:room.value?.rules?.team_size || 3},(_,i)=>i+1));
const allSeated = computed(() => (room.value?.seats?.length || 0) === roomPositions.value.length * 2);
const isLobby = computed(() => ['SEATING', 'READY_CHECK'].includes(room.value?.status));
const draft = computed(() => room.value?.draft || null);
const lineup = computed(() => room.value?.lineup || null);
const settlementGames = computed(() => roomGameKeys.value.map(game => {
  const result = room.value?.match?.results?.find(item => item.game_key === game);
  const key = result?.project_key || gameProject(game);
  return { game_key:game, result, project:room.value?.projects?.find(p=>p.key===key),
    name:key ? projectName(key) : '', icon:roomProjectIcon(key),
    players:{ yellow:lineup.value?.revealed_lineups?.yellow?.[game], white:lineup.value?.revealed_lineups?.white?.[game] } };
}));
const match = computed(() => {
  const value = room.value?.match;
  const packet = localPacket.value;
  const side = value?.my_session?.side;
  if (!value || !packet || value.my_session?.runtime?.instance_id !== packet.instance_id) return value || null;
  const original = value.sessions[side];
  return { ...value, sessions: { ...value.sessions, [side]: { ...original,
    finished: packet.finished || original.finished,
    state: packet.finished ? 'completed' : original.state,
    public_view: { ...original.public_view, sequence: packet.sequence, payload: packet.payload },
  } } };
});
const canPlayLocal = computed(() => Boolean(room.value?.me?.can_move && localRuntime && localPacket.value
  && clockNow.value && localRuntime.playable() && !localPacket.value.finished && projectSyncState.value !== 'blocked'));
const isGameReady = computed(() => /^GAME_[A-O]_READY$/.test(room.value?.status || ''));
const isGamePlaying = computed(() => /^GAME_[A-O]_PLAYING$/.test(room.value?.status || ''));
const isGameResult = computed(() => /^GAME_[A-O]_RESULT$/.test(room.value?.status || ''));
const showPlayingBoards = computed(() => isGamePlaying.value || (isGameResult.value
  && (clockNow.value < finishPlaybackUntil.value || Object.values(observerPending.value).some(Boolean))));
const isGameStage = computed(() => /^GAME_[A-O]_(READY|PLAYING|RESULT)$/.test(room.value?.status || ''));
const remainingSeconds = computed(() => {
  const deadline = Date.parse(
    (room.value?.status === 'LINEUP' ? lineup.value?.deadline_at : draft.value?.deadline_at) || '',
  );
  if (!Number.isFinite(deadline)) return null;
  return Math.max(0, Math.ceil((deadline - clockNow.value) / 1000));
});
const ownRemainingSeconds = computed(() => phaseSeconds(
  ownDeadline(room.value), clockNow.value,
));
const motionEnabled = () => !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const visibleStageMotion = computed(() => stageMotion.value?.to === room.value?.status ? stageMotion.value : null);
const currentGameKey = computed(() => room.value?.match?.current_game_key || null);

function resetStageMotion() {
  window.clearTimeout(stageMotionTimer);
  for (const timer of revealTimers) window.clearTimeout(timer);
  revealTimers = [];
  stageMotion.value = null;
  revealStep.value = 2;
}

function animateLockedIcons(locks) {
  if (!motionEnabled() || !locks.length) return;
  const sources = locks.map(({ key, slot }) => {
    const card = [...document.querySelectorAll('[data-pool-key]')]
      .find((node) => node.dataset.poolKey === key);
    const icon = card?.querySelector('.pool-icon');
    return icon ? { key, slot, rect: icon.getBoundingClientRect(), src: icon.src } : null;
  }).filter(Boolean);
  nextTick(() => {
    for (const source of sources) {
      const target = [...document.querySelectorAll('[data-slot-key]')]
        .find((node) => node.dataset.slotKey === source.slot);
      const targetIcon = target?.querySelector('.history-icon');
      if (!targetIcon || !source.rect.width) continue;
      const end = targetIcon.getBoundingClientRect();
      const ghost = document.createElement('img');
      ghost.src = source.src;
      ghost.alt = '';
      ghost.className = 'lock-flight';
      Object.assign(ghost.style, {
        left: `${source.rect.left}px`, top: `${source.rect.top}px`,
        width: `${source.rect.width}px`, height: `${source.rect.height}px`,
      });
      document.body.append(ghost);
      const dx = end.left - source.rect.left;
      const dy = end.top - source.rect.top;
      const scale = end.width / source.rect.width;
      ghost.animate([
        { transform: 'translate(0, 0) scale(1)', opacity: 1 },
        { transform: `translate(${dx}px, ${dy}px) scale(${scale})`, opacity: .8 },
      ], { duration: 380, easing: 'cubic-bezier(.2,.8,.2,1)', fill: 'forwards' })
        .finished.catch(() => {}).finally(() => ghost.remove());
    }
  });
}

function beginStageMotion(change, next) {
  resetStageMotion();
  if (!change || !motionEnabled()) return;
  const serverNow = Date.parse(next.server_time || '');
  const secondsLeft = phaseSeconds(next.draft?.deadline_at, Number.isFinite(serverNow) ? serverNow : Date.now());
  if ((change.kind === 'first-draw' && secondsLeft !== null && secondsLeft < 3)
    || (change.kind === 'c-draw' && secondsLeft !== null && secondsLeft < 2)) return;
  if (change.kind === 'pick-lock') {
    const locks = change.to === 'SECOND_PICK_BAN'
      ? [{ key: next.draft?.project_a, slot: 'A' }, { key: next.draft?.ban_m, slot: 'M' }]
      : [{ key: next.draft?.project_b, slot: 'B' }, { key: next.draft?.ban_n, slot: 'N' }];
    animateLockedIcons(locks.filter(({ key }) => key));
  }
  stageMotion.value = change;
  if (['first-draw', 'c-draw'].includes(change.kind)) {
    revealStep.value = 0;
    revealTimers.push(window.setTimeout(() => { revealStep.value = 1; }, change.kind === 'first-draw' ? 900 : 430));
    if (change.kind === 'c-draw') revealTimers.push(window.setTimeout(() => { revealStep.value = 2; }, 1050));
  }
  stageMotionTimer = window.setTimeout(() => {
    stageMotion.value = null;
    revealStep.value = 2;
  }, change.kind === 'c-draw' ? 2200 : 1700);
}
const canSubmitPickBan = computed(() => (
  room.value?.me?.can_submit_pick_ban
  && selectedPick.value
  && selectedBan.value
  && selectedPick.value !== selectedBan.value
));
const canSubmitBlind = computed(() => room.value?.me?.can_submit_blind && selectedBlind.value);
const canSubmitLineup = computed(() => (
  room.value?.me?.can_submit_lineup
  && lineupValid(room.value?.rules, lineupSelections.value)
));

function applyRoom(nextRoom, { live = false } = {}) {
  if (!nextRoom?.room_code || nextRoom.room_code.toUpperCase() !== currentCode.value) return false;
  if (room.value?.room_code === nextRoom.room_code) {
    const currentSequence = Number(room.value.event_sequence);
    const nextSequence = Number(nextRoom.event_sequence);
    const currentVersion = Number(room.value.version);
    const nextVersion = Number(nextRoom.version);
    if (Number.isFinite(currentSequence) && Number.isFinite(nextSequence)
      && (nextSequence < currentSequence
        || (nextSequence === currentSequence && nextVersion < currentVersion))) return false;
    if (nextSequence === currentSequence && nextVersion === currentVersion
      && Date.parse(nextRoom.server_time) < Date.parse(room.value.server_time)) return false;
  }
  const phaseChange = stageChange(room.value, nextRoom, live);
  if (isGamePlaying.value && /^GAME_[A-O]_RESULT$/.test(nextRoom.status)) finishPlaybackUntil.value = serverClock.now() + 500;
  if (room.value?.status !== nextRoom.status) beginStageMotion(phaseChange, nextRoom);
  const nextToken = String(nextRoom?.draft?.phase_token || '');
  if (nextToken !== previousPhaseToken) {
    selectedPick.value = '';
    selectedBan.value = '';
    selectedBlind.value = '';
    previousPhaseToken = nextToken;
  }
  const nextLineupToken = String(nextRoom?.lineup?.phase_token || '');
  if (nextLineupToken && nextLineupToken !== previousLineupToken) {
    lineupSelections.value = Object.fromEntries((nextRoom.rules?.game_keys || ['A','B','C']).map((key,i)=>[key,i % (nextRoom.rules?.team_size || 3)+1]));
    previousLineupToken = nextLineupToken;
  }
  if (nextRoom?.lineup?.my_lineup) {
    lineupSelections.value = Object.fromEntries(
      Object.entries(nextRoom.lineup.my_lineup).map(([game, item]) => [game, item.position]),
    );
  }
  const nextResult = nextRoom?.match?.current_result;
  const nextResultKey = nextResult
    ? `${nextResult.game_key}:${nextResult.result_revision}`
    : '';
  if (nextResultKey && nextResultKey !== previousResultKey) {
    overrideYellowScore.value = Number(nextResult.yellow_score || 0);
    overrideWhiteScore.value = Number(nextResult.white_score || 0);
    overrideWinner.value = nextResult.winner_side || 'draw';
    overrideReason.value = '';
    previousResultKey = nextResultKey;
  }
  serverClock.observe(nextRoom.server_time);
  clockNow.value = serverClock.now();
  for (const side of ['yellow', 'white']) {
    const previous = room.value?.match?.sessions?.[side];
    const next = nextRoom.match?.sessions?.[side];
    if (next) next.public_view = receivedProjectView(
      previous?.instance_id === next.instance_id ? previous.public_view : null, next.public_view);
  }
  room.value = nextRoom;
  synchronizeRuntime(nextRoom);
  return true;
}

function localStorageKey(instanceId) {
  return `competition:runtime:${session.value?.user?.id}:${instanceId}`;
}
function saveLocalState() {
  window.clearTimeout(localSaveTimer);
  if (!localPacket.value) return;
  try { window.sessionStorage.setItem(localStorageKey(localPacket.value.instance_id), JSON.stringify(localPacket.value)); }
  catch { /* Storage may be disabled; the server still keeps the latest checkpoint. */ }
}
function disposeRuntime() {
  saveLocalState();
  stateSender?.close(); stateSender = null;
  localRuntime = null; localPacket.value = null;
  projectSyncState.value = 'synced';
  window.clearTimeout(projectThinkingTimer);
  projectThinking.value = false; movePending.value = false;
}
function commitLocal(packet) {
  if (!packet) return;
  localPacket.value = packet;
  projectSyncState.value = 'syncing';
  window.clearTimeout(localSaveTimer);
  localSaveTimer = window.setTimeout(saveLocalState, 100);
  stateSender?.push(packet);
}
function synchronizeRuntime(nextRoom) {
  const own = nextRoom?.match?.my_session;
  const bootstrap = own?.runtime;
  if (!bootstrap || !/^GAME_[A-O]_PLAYING$/.test(nextRoom.status)) {
    disposeRuntime();
    return;
  }
  const side = own.side;
  const clock = nextRoom.match.sessions[side].project_clock;
  if (localRuntime?.bootstrap.instance_id !== bootstrap.instance_id) {
    disposeRuntime();
    let cached = null;
    try { cached = JSON.parse(window.sessionStorage.getItem(localStorageKey(bootstrap.instance_id)) || 'null'); } catch { /* No saved state. */ }
    if (cached?.instance_id !== bootstrap.instance_id || cached.sequence < bootstrap.sequence) cached = null;
    const config = cached ? { ...bootstrap, checkpoint: cached.checkpoint, sequence: cached.sequence } : bootstrap;
    try {
      localRuntime = new MatchRuntime(config, { elapsedMs: clock.elapsed_ms, running: clock.running });
    } catch (cause) {
      projectSyncState.value = 'blocked'; setError(cause); return;
    }
    acknowledgedSequence = bootstrap.sequence;
    const instance = bootstrap.instance_id;
    const code = nextRoom.room_code;
    stateSender = new ProjectStreamSender({
      createSocket: () => projectSocket(code),
      authenticate: () => projectAuthentication(instance),
      phaseToken: () => room.value?.match?.phase_token,
      onAck: ack => {
        if (localRuntime?.bootstrap.instance_id !== instance) return;
        acknowledgedSequence = Math.max(acknowledgedSequence, ack.accepted_sequence || 0);
        if (ack.accepted_sequence > localRuntime.sequence) disconnectRoom?.resync?.();
        projectSyncState.value = acknowledgedSequence >= (localPacket.value?.sequence || 0) ? 'synced' : 'syncing';
        if (ack.competition) applyRoom(ack.competition, { live: true });
        else if (ack.update) applyProjectUpdate(ack.update);
        if (ack.resync_room || ack.stopped) disconnectRoom?.resync?.();
      },
      onError: cause => {
        if (localRuntime?.bootstrap.instance_id !== instance) return false;
        projectSyncState.value = 'reconnecting';
        if (['STALE_PHASE', 'MATCH_SUSPENDED'].includes(cause.code)) {
          void refreshRoom();
          return true;
        }
        if (cause.code === 'STREAM_DISCONNECTED') void refreshRoom();
        if ([400, 401, 403, 404, 413, 422, 426].includes(cause.status)) {
          projectSyncState.value = 'blocked'; setError(cause); return false;
        }
        if (cause.code === 'TEAM_CLOCK_EXPIRED') return false;
        return true;
      },
    });
    if (!config.checkpoint) commitLocal(localRuntime.accept());
    else {
      localPacket.value = localRuntime.packet();
      if (cached && cached.sequence > bootstrap.sequence) commitLocal(localPacket.value);
    }
  } else if (bootstrap.sequence > localRuntime.sequence && bootstrap.checkpoint) {
    localRuntime.restore(bootstrap.checkpoint);
    localRuntime.sequence = bootstrap.sequence;
    localPacket.value = localRuntime.packet();
  }
  localRuntime?.setClock(clock.elapsed_ms, clock.running);
  if (own.finished && !localPacket.value?.finished) {
    // Clock expiry / referee decisions remain authoritative match controls.
    localRuntime.game.finished = true;
    localRuntime.game.outcome = own.outcome;
    localRuntime.frozenElapsed = clock.elapsed_ms;
    localPacket.value = localRuntime.packet();
  }
  if (bootstrap.race_stop_requested && !movePending.value) commitLocal(localRuntime?.stopRace());
}

function applyProjectUpdate(update) {
  const current = room.value;
  const state = current?.match?.sessions?.[update.side];
  if (!state || current.room_code !== update.room_code || current.match.current_game_key !== update.game_key
    || state.instance_id !== update.instance_id || !/^GAME_[A-O]_PLAYING$/.test(current.status)) return;
  // A full checkpoint may skip intermediate frames. Never animate a move from
  // a board that the receiver did not actually display.
  const previousSequence = Number(state.public_view?.sequence || 0);
  const view = receivedProjectView(state.public_view, update.public_view);
  state.public_view = view;
  if (Number(update.public_view.sequence) === Number(view.sequence) && Number(view.sequence) > previousSequence) {
    state.project_clock = { ...state.project_clock, elapsed_ms: view.payload.elapsed_ms, sampled_at: update.server_time };
  }
  current.version = Math.max(current.version, update.version);
}

function setError(cause) {
  error.value = userFacingError(cause);
  if (cause?.code === 'REMOVED_FROM_ROOM') {
    disposeRuntime();
    disconnectRoom?.();
    disconnectRoom = null;
    room.value = null;
    connection.value = 'offline';
  }
}

async function refreshRoom() {
  const code = currentCode.value, epoch = routeEpoch;
  if (!code || roomRefreshPending) return;
  roomRefreshPending = true;
  try {
    const data = await api.room(code, { timeoutMs: 5000 });
    if (epoch === routeEpoch && currentCode.value === code) applyRoom(data.competition, { live: true });
  } catch { /* The socket retry continues while the network is unavailable. */ }
  finally { roomRefreshPending = false; }
}

function recoverConnections() {
  if (document.hidden) return;
  clockNow.value = serverClock.now();
  disconnectRoom?.resync?.();
  stateSender?.recover();
  void refreshRoom();
}

async function loadDashboard(epoch) {
  disposeRuntime();
  resetStageMotion();
  room.value = null;
  disconnectRoom?.();
  disconnectRoom = null;
  connection.value = 'offline';
  const [payload, directory] = await Promise.all([
    session.value ? api.list() : Promise.resolve({ competitions: [] }), api.events(),
  ]);
  if (epoch !== routeEpoch || currentCode.value) return;
  rooms.value = payload.competitions || [];
  events.value = directory.events || [];
  canCreateEvent.value = directory.can_create;
}

async function loadRoom(code, epoch) {
  disposeRuntime();
  resetStageMotion();
  disconnectRoom?.();
  disconnectRoom = null;
  room.value = null;
  connection.value = 'connecting';
  const payload = await api.checkIn(code);
  if (epoch !== routeEpoch || currentCode.value !== code) return;
  if (!applyRoom(payload.competition)) throw new Error('房间响应与当前地址不匹配。');
  disconnectRoom = connectRoom(code, {
    onClose: (event) => {
      if (epoch !== routeEpoch) return;
      connection.value = [4401, 4403, 4404, 1008].includes(event?.code) ? 'offline' : 'reconnecting';
      if (event?.code === 4401) setError('房间连接未通过身份验证，请重新登录。');
    },
    onError: () => { if (epoch === routeEpoch) connection.value = 'reconnecting'; },
    onMessage: (message) => {
      if (epoch !== routeEpoch || currentCode.value !== code) return;
      if (message?.type === 'room.snapshot') {
        applyRoom(message.data, { live: connection.value === 'online' });
        connection.value = 'online';
      }
      if (message?.type === 'project.snapshot') applyProjectUpdate(message.data);
      if (['error', 'room.error'].includes(message?.type)) setError(message.error || '房间连接失败');
    },
  });
}

async function route() {
  const epoch = ++routeEpoch;
  loading.value = true;
  error.value = '';
  try {
    if (isProjectRoute.value) {
      disposeRuntime();
      disconnectRoom?.();
      disconnectRoom = null;
      room.value = null;
      connection.value = 'offline';
      return;
    }
    if (!session.value) {
      const payload = await api.session().catch(cause => {
        if (cause.status === 401 && !currentCode.value) return null;
        throw cause;
      });
      if (epoch !== routeEpoch) return;
      session.value = payload;
    }
    if (currentCode.value) await loadRoom(currentCode.value, epoch);
    else await loadDashboard(epoch);
  } catch (cause) {
    if (epoch === routeEpoch) setError(cause);
  } finally {
    if (epoch === routeEpoch) loading.value = false;
  }
}

function navigate(path) {
  window.history.pushState({}, '', path);
  pathname.value = window.location.pathname;
  route();
}

function handleLocationChange() {
  pathname.value = window.location.pathname;
  route();
}

async function perform(action, { ignoreCodes = [] } = {}) {
  if (busy.value) return false;
  busy.value = true;
  error.value = '';
  try {
    const payload = await action();
    if (payload?.competition) applyRoom(payload.competition, { live: true });
    return true;
  } catch (cause) {
    if (!ignoreCodes.includes(cause?.code)) setError(cause);
    return cause?.code || false;
  } finally {
    busy.value = false;
  }
}

async function createRoom() {
  await perform(async () => {
    const projects = selectedProjects.value;
    if (!newRoomRulesValid.value) throw new Error(language.value==='zh'?'请检查房间规则及项目池。':'Check the room rules and project pool.');
    if (new Set(projects.map(project => project.id)).size !== projects.length) {
      throw new Error('项目池不能包含重复项目。');
    }
    const payload = await api.create(
      newRoomName.value,
      projects.map(competitionProjectInput),
      newRoomEventSlug.value || null,
      {...newRoomSchedule.value, rules:newRoomRules.value},
    );
    navigate(`/rooms/${payload.competition.room_code}`);
    return payload;
  });
}

async function closeRoom(item) {
  if (busy.value || !(item?.can_close ?? item?.me?.can_close)) return;
  if (!window.confirm(t(`确定关闭「${item.name}」？关闭后无法重新落座或开赛，房间记录仍会保留。`))) return;
  busy.value = true;
  error.value = '';
  try {
    await api.closeRoom(item.room_code);
    if (currentCode.value === item.room_code) navigate(competitionHomePath);
    else rooms.value = (await api.list()).competitions || [];
  } catch (cause) {
    setError(cause);
  } finally {
    busy.value = false;
  }
}

function moveProjectOrder(id, offset) {
  const index = selectedProjectIds.value.indexOf(id);
  const target = index + offset;
  if (index < 0 || target < 0 || target >= selectedProjectIds.value.length) return;
  const updated = [...selectedProjectIds.value];
  [updated[index], updated[target]] = [updated[target], updated[index]];
  selectedProjectIds.value = updated;
}

function enterRoom() {
  const code = joinCode.value.trim().toUpperCase();
  if (code) navigate(`/rooms/${encodeURIComponent(code)}`);
}

function claim(side, position) {
  perform(() => api.claimSeat(room.value.room_code, side, position));
}

function leave() {
  perform(() => api.leaveSeat(room.value.room_code));
}

function toggleReady() {
  perform(() => api.ready(room.value.room_code, !myTeamReady.value));
}

function choosePick(key) {
  selectedPick.value = selectedPick.value === key ? '' : key;
  if (selectedBan.value === key) selectedBan.value = '';
}

function chooseBan(key) {
  selectedBan.value = selectedBan.value === key ? '' : key;
  if (selectedPick.value === key) selectedPick.value = '';
}

function submitPickBan() {
  if (!canSubmitPickBan.value) return;
  perform(() => api.submitPickBan(
    room.value.room_code,
    selectedPick.value,
    selectedBan.value,
    draft.value.phase_token,
  ));
}

function submitBlind() {
  if (!canSubmitBlind.value) return;
  perform(() => api.submitBlind(
    room.value.room_code,
    selectedBlind.value,
    draft.value.phase_token,
  ));
}

function submitLineup() {
  if (!canSubmitLineup.value) return;
  perform(() => api.submitLineup(
    room.value.room_code,
    Object.fromEntries(
      Object.entries(lineupSelections.value).map(([game, position]) => [game, Number(position)]),
    ),
    lineup.value.phase_token,
  ));
}

function toggleGameReadiness(role) {
  const side = mySeat.value?.side;
  if (!side || !match.value) return;
  const field = role === 'player' ? 'player_ready' : 'captain_ready';
  perform(() => api.setGameReadiness(
    room.value.room_code,
    role,
    !match.value.readiness?.[side]?.[field],
    match.value.phase_token,
  ));
}

async function moveGame(direction) {
  if (!localRuntime || !canPlayLocal.value || movePending.value) return;
  const runtime = localRuntime;
  try {
    const result = runtime.move(direction);
    if (!result?.then) { commitLocal(result); return; }
    movePending.value = true;
    projectThinkingTimer = window.setTimeout(() => { if (movePending.value) projectThinking.value = true; }, 300);
    const packet = await result;
    if (localRuntime === runtime) {
      commitLocal(packet);
      if (room.value?.match?.my_session?.runtime?.race_stop_requested) commitLocal(runtime.stopRace());
    }
  } catch (cause) { setError(`本机 AI 计算失败，请重试：${cause.message || cause}`); }
  finally {
    if (localRuntime === runtime) {
      window.clearTimeout(projectThinkingTimer); projectThinking.value = false; movePending.value = false;
    }
  }
}

function projectAction(action) {
  if (action === 'surrender' && (!canSurrender.value || !window.confirm(t('确认认输本局？将保留当前得分并判本方负，无法撤回。')))) return;
  if (localRuntime && canPlayLocal.value && !movePending.value) commitLocal(localRuntime.action(action));
}

const canSurrender = computed(() => {
  const side = mySeat.value?.side;
  const other = side === 'yellow' ? 'white' : 'yellow';
  return canPlayLocal.value && !!match.value?.sessions?.[other]?.finished && !match.value?.sessions?.[side]?.finished;
});
const stageWait = computed(() => {
  const deadline = isGameResult.value ? match.value?.rest_until : match.value?.preview_until;
  return Math.max(0, Math.ceil((Date.parse(deadline || '') - clockNow.value) / 1000)) || 0;
});
function manageMember(userId, remove = true) {
  if (!Number(userId) || !window.confirm(t(remove ? '移出此人并禁止其重新进入本房间？赛中会暂停并保留历史记录。' : '允许此人重新进入？比赛不会自动解除暂停。'))) return;
  perform(() => api.manageMember(room.value.room_code, Number(userId), remove));
}
function projectMark(key) {
  if (draft.value?.project_a === key) return '项目 A';
  if (draft.value?.project_b === key) return '项目 B';
  if (draft.value?.ban_m === key) return `${sideName(draft.value.first_side)} BAN`;
  if (draft.value?.ban_n === key) return `${sideName(draft.value.second_side)} BAN`;
  return '';
}

function confirmResult() {
  if (!room.value?.me?.can_confirm_result) return;
  perform(() => api.confirmCurrentResult(
    room.value.room_code,
    match.value.current_result.result_revision,
    match.value.phase_token,
  ));
}

function reportIssue() {
  if (!room.value?.me?.can_report_issue || issueDetails.value.trim().length < 3) return;
  perform(() => api.reportIssue(
    room.value.room_code,
    issueCategory.value,
    issueDetails.value.trim(),
  ));
}

function resolveIssue(issue, status = 'resolved') {
  if (issueResolutionNote.value.trim().length < 3) return;
  perform(() => api.resolveIssue(
    room.value.room_code,
    issue.id,
    status,
    issueResolutionNote.value.trim(),
  ));
}

function suspendMatch() {
  if (!room.value?.me?.can_suspend || suspendReasonText.value.trim().length < 3) return;
  perform(() => api.suspendMatch(
    room.value.room_code,
    suspendReasonCode.value,
    suspendReasonText.value.trim(),
    match.value.phase_token,
  ));
}

function toggleResumeReadiness() {
  const side = mySeat.value?.side;
  if (!side || !match.value?.suspension?.active) return;
  perform(() => api.setSuspensionReadiness(
    room.value.room_code,
    !match.value.suspension.resume_readiness?.[side]?.ready,
    match.value.phase_token,
  ));
}

function resumeMatch() {
  if (!room.value?.me?.can_resume) return;
  perform(() => api.resumeMatch(room.value.room_code, match.value.phase_token));
}

function overrideResult() {
  if (!room.value?.me?.can_override_result || overrideReason.value.trim().length < 3) return;
  perform(() => api.overrideCurrentResult(
    room.value.room_code,
    Number(overrideYellowScore.value),
    Number(overrideWhiteScore.value),
    overrideWinner.value,
    overrideReason.value.trim(),
    match.value.current_result.result_revision,
    match.value.phase_token,
  ));
}

function forceAdvance() {
  if (!room.value?.me?.can_force_advance || forceAdvanceReason.value.trim().length < 3) return;
  perform(() => api.forceAdvanceCurrentResult(
    room.value.room_code,
    forceAdvanceReason.value.trim(),
    match.value.phase_token,
  ));
}

function forceFinish() {
  if (!room.value?.me?.can_force_finish || forceFinishReason.value.trim().length < 3) return;
  perform(() => api.forceFinishMatch(
    room.value.room_code,
    forceFinishWinner.value,
    forceFinishReason.value.trim(),
    match.value.phase_token,
  ));
}

function projectName(key) {
  return t(room.value?.projects?.find((project) => project.key === key)?.name || key || '—');
}

function projectDescription(key) {
  return room.value?.projects?.find((project) => project.key === key)?.description || '具体目标与判定以本场公布的项目规则为准。';
}

function roomProjectIcon(key) {
  return projectIconUrl(room.value?.projects?.find((project) => project.key === key)?.project_ref);
}

function sessionPayload(side) {
  return match.value?.sessions?.[side]?.public_view?.payload || null;
}

function setObserverPending(side, pending) {
  const wasPending = observerPending.value[side];
  observerPending.value[side] = pending;
  if (wasPending && !pending && isGameResult.value) finishPlaybackUntil.value = Math.max(finishPlaybackUntil.value, serverClock.now() + 300);
}

function projectMetric(side) {
  return projectPerformanceMetric(match.value?.sessions?.[side]?.public_view, language.value);
}
function ruleMetrics(side) {
  const game = match.value?.games?.find(item => item.game_key === match.value.current_game);
  const project = room.value?.projects?.find(item => item.key === game?.project_key) || {};
  return projectRuleMetrics(match.value?.sessions?.[side]?.public_view, language.value, project);
}
function projectResultValue(result, side) { return rawProjectResultValue(result, side, language.value); }

function projectBoardSnapshot(side) {
  const view = match.value?.sessions?.[side]?.public_view;
  const payload = view?.payload || {};
  const board = Array.isArray(payload.board?.[0]) ? payload.board.flat() : (payload.board || []);
  return {
    ...payload,
    board,
    rows: Number(payload.rows || Math.sqrt(board.length) || 4),
    cols: Number(payload.cols || Math.sqrt(board.length) || 4),
    revision: Number(view?.sequence || payload.move_count || 0),
    transition: payload.last_transition || null,
  };
}

function projectElapsedMs(side) {
  if (localRuntime && localPacket.value && isMyActiveSide(side)) {
    void clockNow.value;
    return localRuntime.elapsed();
  }
  const clock = match.value?.sessions?.[side]?.project_clock;
  const elapsedBase = clock?.elapsed_ms ?? sessionPayload(side)?.elapsed_ms ?? 0;
  let elapsed = Math.max(0, Number(elapsedBase));
  const snapshotAt = Date.parse(clock?.sampled_at || room.value?.server_time || '');
  if (clock?.running && Number.isFinite(snapshotAt)) {
    elapsed += Math.max(0, clockNow.value - snapshotAt);
  }
  return elapsed;
}

function projectRemainingMs(side) {
  const clock = match.value?.sessions?.[side]?.project_clock;
  const limit = Number(clock?.limit_ms ?? sessionPayload(side)?.time_limit_ms ?? 0);
  return limit > 0 ? Math.max(0, limit - projectElapsedMs(side)) : null;
}

function rosterStatus(side, position) {
  if (Number(match.value?.players?.[side]?.position) !== position) return '候场';
  if (isGameReady.value) {
    return match.value?.readiness?.[side]?.player_ready ? '已就绪' : '待就绪';
  }
  if (isGameResult.value || match.value?.sessions?.[side]?.finished) return '本场完成';
  return '出战中';
}

function formatProjectElapsed(milliseconds) {
  const value = Math.max(0, Number(milliseconds) || 0);
  const minutes = Math.floor(value / 60000).toString().padStart(2, '0');
  const seconds = (Math.floor(value / 1000) % 60).toString().padStart(2, '0');
  const centiseconds = (Math.floor(value / 10) % 100).toString().padStart(2, '0');
  return `${minutes}:${seconds}.${centiseconds}`;
}

function isMyActiveSide(side) {
  return match.value?.my_session?.side === side;
}

function canUseBoard(side) {
  return isMyActiveSide(side) && !movePending.value
    && canPlayLocal.value;
}

function sideName(side) {
  return t(side === 'yellow' ? '黄方' : side === 'white' ? '白方' : '—');
}

function captainName(side) {
  return room.value?.seats?.find((seat) => seat.side === side && seat.position === 1)?.display_name || '队长';
}

function playerAt(side, position) {
  return room.value?.seats?.find(
    (seat) => seat.side === side && seat.position === Number(position),
  )?.display_name || ` ${position} 号位`;
}

function gameProject(game) {
  return draft.value?.workflow?.selected?.[roomGameKeys.value.indexOf(game)] || draft.value?.[`project_${String(game).toLowerCase()}`] || '';
}

function draftSource(key) {
  const source = draft.value?.sources?.[key];
  return source === 'timeout' ? '超时自动' : source === 'captain' ? '队长锁定' : '';
}

function resultForGame(game) {
  return match.value?.results?.find((result) => result.game_key === game) || null;
}

function isProjectAvailable(key) {
  return draft.value?.available_project_keys?.includes(key);
}

function formatCountdown(value) {
  if (value === null) return '--:--';
  const minutes = Math.floor(value / 60).toString().padStart(2, '0');
  const seconds = (value % 60).toString().padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function teamClockMs(side) {
  if (localRuntime && isMyActiveSide(side) && localPacket.value?.finished) {
    return Math.max(0, localRuntime.budget() - localRuntime.elapsed());
  }
  const clock = match.value?.clocks?.[side];
  if (!clock) return 0;
  const deadline = Date.parse(clock.deadline_at || '');
  if (clock.state === 'running' && Number.isFinite(deadline)) {
    return Math.max(0, deadline - clockNow.value);
  }
  return Math.max(0, Number(clock.remaining_ms || 0));
}

function formatTeamClock(milliseconds) {
  const total = Math.max(0, Math.ceil(milliseconds / 1000));
  const minutes = Math.floor(total / 60).toString().padStart(2, '0');
  const seconds = (total % 60).toString().padStart(2, '0');
  return `${minutes}:${seconds}`;
}

function winnerName(side) {
  if (side === 'draw') return '平局';
  return `${sideName(side)}胜`;
}

function handleGameKeys(event) {
  if (!canPlayLocal.value || movePending.value) return;
  if (event.defaultPrevented || event.isComposing || event.keyCode === 229) return;
  const target = event.target;
  if (target instanceof Element && target.closest('input, textarea, select, [contenteditable]')) return;
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  if (['r', 'z'].includes(event.key.toLowerCase())) {
    event.preventDefault();
    if (!event.repeat) projectAction(event.key.toLowerCase() === 'r' ? 'restart' : 'undo');
    return;
  }
  const direction = {
    ArrowUp: 'up', w: 'up', W: 'up',
    ArrowDown: 'down', s: 'down', S: 'down',
    ArrowLeft: 'left', a: 'left', A: 'left',
    ArrowRight: 'right', d: 'right', D: 'right',
  }[event.key];
  if (!direction) return;
  event.preventDefault();
  moveGame(direction);
}

function seatLabel(side, position) {
  return `${side === 'yellow' ? '黄' : '白'}${position}`;
}

async function rematchRoom() {
  if(busy.value || !window.confirm(t('因落位错误重赛？原房间取消、下注退还，新建空席位房间。已过开战时间时，从现在重新计算15分钟就位期限。')))return;
  busy.value=true;
  try{const result=await api.rematch(room.value.room_code);navigate(`/rooms/${result.competition.room_code}`);}catch(cause){setError(cause);}finally{busy.value=false;}
}
function canClaim(seat,side,position) {
  const players=room.value?.schedule?.players || [];
  return !seat && room.value?.me?.can_claim_seat && !busy.value && (!players.length || players.some(p=>p.user_id===room.value.me.user_id && p.side===side && p.position===position));
}

function teamSeats(side) {
  return roomPositions.value.map((position) => ({
    side,
    position,
    seat: seats.value.get(`${side}:${position}`),
  }));
}

onMounted(() => {
  window.addEventListener('popstate', handleLocationChange);
  window.addEventListener('keydown', handleGameKeys);
  ticker = window.setInterval(() => {
    clockNow.value = serverClock.now();
    if (localRuntime && !movePending.value) commitLocal(localRuntime.tick());
  }, 16);
  window.addEventListener('pagehide', saveLocalState);
  window.addEventListener('online', recoverConnections);
  document.addEventListener('visibilitychange', recoverConnections);
  route();
});

onBeforeUnmount(() => {
  disposeRuntime();
  window.removeEventListener('pagehide', saveLocalState);
  window.removeEventListener('online', recoverConnections);
  document.removeEventListener('visibilitychange', recoverConnections);
  ++routeEpoch;
  window.removeEventListener('popstate', handleLocationChange);
  window.removeEventListener('keydown', handleGameKeys);
  window.clearInterval(ticker);
  window.clearTimeout(projectThinkingTimer);
  resetStageMotion();
  disconnectRoom?.();
});
</script>

<template>
  <ProjectPlayground v-if="isProjectRoute" :project-id="projectRouteId" />
  <div v-else :class="['app-shell', isGameStage && 'active-game-shell']">
    <header class="site-header">
      <button class="brand" type="button" @click="navigate(competitionHomePath)">
        <span class="brand-mark">20</span>
        <span><strong>{{ $t("2048 赛事中心") }}</strong><small>Competition</small></span>
      </button>
      <div v-if="session?.user" class="account-chip">
        <span class="account-dot"></span>
        {{ session.user.display_name }}
      </div>
      <LanguageSwitch />
    </header>

    <main v-if="loading" class="center-state">
      <span class="spinner"></span>
      <p>{{ $t("正在同步比赛状态…") }}</p>
    </main>

    <main v-else-if="error && !session && currentCode" class="center-state error-state">
      <h1>{{ $t("需要登录") }}</h1>
      <p>{{ $t(error) }}</p>
      <a class="primary-button" :href="mainSiteUrl">{{ $t("前往主站登录") }}</a>
    </main>

    <main v-else-if="!room" class="dashboard page-width">
      <EventCenter :key="eventSlug" :slug="eventSlug" :events="events" :can-create="canCreateEvent" :status-text="statusText" @navigate="navigate" @refresh="loadDashboard(routeEpoch)" @create-room="slug => { newRoomEventSlug = slug; createRoomOpen = true; }" />
      <section v-if="eventHasRooms" class="hero-row">
        <div>
          <h2>{{ $t("快捷进入房间") }}</h2>
          <p class="muted">{{ $t("已有房间码？登录后进入候场。观众请从") }}<a href="https://live.2048tables.online/" target="_blank" rel="noopener noreferrer">{{ $t("直播大厅") }}</a>{{ $t("进入。") }}</p>
        </div>
        <form class="join-box" @submit.prevent="enterRoom">
          <label for="room-code">{{ $t("房间码") }}</label>
          <div class="inline-form">
            <input id="room-code" v-model="joinCode" maxlength="12" :placeholder='$t("例如 K8F3QX")' />
            <button class="primary-button" type="submit">{{ $t("进入房间") }}</button>
          </div>
        </form>
      </section>

      <p v-if="error" class="alert">{{ $t(error) }}</p>

      <details v-if="eventHasRooms && session && (session.can_create_competition || events.some(e => e.can_manage && e.capabilities.rooms))" :open="createRoomOpen" class="panel room-create-disclosure" @toggle="createRoomOpen = $event.target.open">
        <summary>{{ $t("创建比赛房间") }}</summary>
        <div>
          <p class="eyebrow">{{ $t("举办方") }}</p>
          <h2>{{ $t("创建新比赛") }}</h2>
          <p class="muted">{{ language==='zh'?'选择比赛流程与项目池；创建后规则与顺序固定。':'Choose a match format and project pool. Rules and order are frozen on creation.' }}</p>
        </div>
        <form class="create-form" @submit.prevent="createRoom">
          <label class="create-name">{{ $t("所属赛事") }}<select v-model="newRoomEventSlug"><option value="">{{ $t("独立 / 测试房间") }}</option><option v-for="event in events.filter(item => item.can_manage && item.status !== 'finished' && item.capabilities.rooms)" :key="event.slug" :value="event.slug">{{ event.name }}</option></select></label>
          <label class="create-name">{{ $t("比赛名称") }}<input v-model="newRoomName" minlength="2" maxlength="100" required :placeholder='$t("比赛名称")' /></label>
          <RoomRulesPicker :pool-size="selectedProjects.length" @change="({rules,valid}) => {newRoomRules=rules;newRoomRulesValid=valid;}" />
          <RoomSchedulePicker :team-size="newRoomRules?.team_size || 3" :slug="newRoomEventSlug" @change="newRoomSchedule=$event" />
          <fieldset class="project-picker">
            <legend>{{ $t("项目池 · 已选 ") }}{{ $t(selectedProjects.length) }}{{ $t(" 项") }}</legend>
            <p class="muted">{{ $t("勾选项目后，可调整其在 BP 项目池中的顺序。") }}</p>
            <div v-for="project in orderedProjectOptions" :key="project.id" class="project-picker-row">
              <label><input v-model="selectedProjectIds" type="checkbox" :value="project.id" /><img class="project-icon picker-icon" :src="projectIconUrl(project.id)" alt="" /><span><strong>{{ $t(project.title) }}</strong><small>{{ $t(project.description) }}</small></span></label>
              <div class="project-picker-actions">
                <a :href="project.practicePath" target="_blank" rel="noopener noreferrer">{{ $t("试玩") }}</a>
                <button type="button" :disabled="!selectedProjectIds.includes(project.id) || selectedProjectIds.indexOf(project.id) === 0" :aria-label="$t(`上移 ${project.title}`)" @click="moveProjectOrder(project.id, -1)">↑</button>
                <button type="button" :disabled="!selectedProjectIds.includes(project.id) || selectedProjectIds.indexOf(project.id) === selectedProjectIds.length - 1" :aria-label="$t(`下移 ${project.title}`)" @click="moveProjectOrder(project.id, 1)">↓</button>
                <span v-if="selectedProjectIds.includes(project.id)" class="project-picker-order">{{ $t(selectedProjectIds.indexOf(project.id) + 1) }}</span>
              </div>
            </div>
          </fieldset>
          <p v-if="!newRoomRulesValid" class="alert">{{ language==='zh'?'请先完善比赛流程与项目池。':'Complete the match format and project pool first.' }}</p>
          <button class="primary-button" type="submit" :disabled="busy || !newRoomRulesValid">{{ $t("创建房间") }}</button>
        </form>
      </details>

      <section v-if="session && eventHasRooms" class="room-list-section">
        <div class="section-heading"><h2>{{ $t("我的比赛房间") }}</h2><span>{{ $t(activeRooms.length) }}{{ $t(" 场") }}</span></div>
        <div v-if="activeRooms.length" class="room-list">
          <div v-for="item in activeRooms" :key="item.id" class="room-row">
            <button class="room-row-link" type="button" @click="navigate(`/rooms/${item.room_code}`)">
              <span><strong>{{ roomMatchTitle(item) }} <b>{{ $t(roomMatchScore(item)) }}</b></strong><small>{{ item.event?.name || $t('独立房间') }} · {{ $t(item.room_code) }} · {{ $t(scheduleTime(item.schedule?.starts_at)) }}{{ $t(item.schedule ? '（北京时间）' : '') }}</small></span>
              <span class="room-meta">{{ $t(item.schedule?.exception === 'both_late' ? '双方未就位 · 0:0' : statusText[item.status] || item.status) }}</span>
            </button>
            <button v-if="item.can_close" class="room-close-button" type="button" :disabled="busy" :aria-label="`关闭比赛房间 ${item.name}`" @click="closeRoom(item)">{{ $t("关闭房间") }}</button>
          </div>
        </div>
        <p v-else class="empty-state">{{ $t("暂无进行中的比赛房间。") }}</p>
        <details v-if="closedRooms.length" class="closed-room-list">
          <summary>{{ $t("已关闭的房间 · ") }}{{ $t(closedRooms.length) }}{{ $t(" 场") }}</summary>
          <div class="room-list">
            <button v-for="item in closedRooms" :key="item.id" class="room-row room-row-link" type="button" @click="navigate(`/rooms/${item.room_code}`)">
              <span><strong>{{ roomMatchTitle(item) }} {{ $t(roomMatchScore(item)) }}</strong><small>{{ $t(item.room_code) }} · {{ $t(scheduleTime(item.schedule?.starts_at)) }}</small></span>
              <span class="room-meta">{{ $t("房间已关闭 · 查看记录") }}</span>
            </button>
          </div>
        </details>
      </section>
    </main>

    <main v-else :class="['room-page', isGameStage && 'game-room']">
      <section class="match-header">
        <div class="page-width match-header-inner">
          <div class="match-header-actions">
            <button class="back-button" type="button" @click="navigate(room.event ? `/events/${room.event.slug}` : competitionHomePath)">← {{ room.event?.name || $t('赛事中心') }}</button>
            <button v-if="room.me.can_close" class="close-room-link" type="button" :disabled="busy" @click="closeRoom(room)">{{ $t("关闭房间") }}</button>
            <button v-if="(room.me.can_manage || room.me.staff_roles?.includes('referee')) && ['SEATING','READY_CHECK','DRAW','DRAFT_STEP','FIRST_PICK_BAN','SECOND_PICK_BAN','BLIND_PICK','C_DRAW'].includes(room.status)" class="close-room-link" :disabled="busy" @click="rematchRoom">{{ $t("落位错误重赛") }}</button>
          </div>
          <div class="match-identity">
            <span class="room-code">{{ $t(room.room_code) }}</span>
            <h1>{{ room.name }}</h1>
          </div>
          <div class="stage-chip"><span :class="['connection-dot', connection]"></span>{{ $t(match?.suspension?.active ? '暂停中 · ' : '') }}{{ $t(statusText[room.status] || room.status) }}</div>
        </div>
      </section>

      <section v-if="!['CANCELLED','FINISHED'].includes(room.status)" class="progress-strip">
        <div :class="['progress-step', room.status === 'SEATING' ? 'active' : 'done']">{{ $t("选手落座") }}</div>
        <div :class="['progress-step', room.status === 'READY_CHECK' ? 'active' : !['SEATING', 'READY_CHECK'].includes(room.status) ? 'done' : '']">{{ $t("队长准备") }}</div>
        <div :class="['progress-step', room.status === 'DRAW' ? 'active' : !['SEATING', 'READY_CHECK', 'DRAW'].includes(room.status) ? 'done' : '']">{{ $t("先后手抽签") }}</div>
        <div :class="['progress-step', ['DRAFT_STEP', 'FIRST_PICK_BAN', 'SECOND_PICK_BAN', 'BLIND_PICK'].includes(room.status) ? 'active' : ['C_DRAW', 'LINEUP'].includes(room.status) || room.status.startsWith('GAME_') || room.status === 'FINISHED' ? 'done' : '']">{{ $t("项目 BP") }}</div>
        <div :class="['progress-step', room.status === 'LINEUP' ? 'active' : room.status.startsWith('GAME_') || room.status === 'FINISHED' ? 'done' : '']">{{ $t("秘密布阵") }}</div>
        <div :class="['progress-step', room.status.startsWith('GAME_') ? 'active' : room.status === 'FINISHED' ? 'done' : '']">{{ $t("项目对局") }}</div>
      </section>

      <div class="page-width room-content">
        <details v-if="room.rules && !isGameStage" class="frozen-room-rules">
          <summary>{{ language==='zh'?'本房间规则':'Room rules' }} · {{ room.rules.game_count }} {{ language==='zh'?'局':'games' }} · {{ room.rules.team_size }} v {{ room.rules.team_size }}</summary>
          <p>{{ room.rules.series_mode==='all'?(language==='zh'?'打满全部对局':'Play every game'):(language==='zh'?`先赢 ${room.rules.wins_required} 局`:`First to ${room.rules.wins_required} wins`) }} · {{ lineupPolicyDescription(room.rules.lineup_policy, language) }}</p>
          <p>BP {{ room.rules.draft_seconds }}s · {{ language==='zh'?'布阵':'Lineup' }} {{ room.rules.lineup_seconds }}s · {{ language==='zh'?'每队总计时':'Team clock' }} {{ room.rules.team_clock_seconds }}s</p>
          <p class="frozen-steps"><span v-for="(step,i) in room.rules.steps" :key="i">{{ i+1 }}. {{ step.actor==='first'?(language==='zh'?'先手':'First'):(language==='zh'?'后手':'Second') }} B{{ step.bans }} P{{ step.picks }}</span><span>{{ room.rules.final_selection==='blind'?(language==='zh'?'双方盲选后抽签':'Draw between blind picks'):(language==='zh'?'剩余项目池抽签':'Draw from remaining pool') }}</span></p>
        </details>
        <section v-if="room.schedule && (isLobby || match?.finish_reason === 'late_forfeit')" class="panel schedule-room-notice">
          <h2>{{ roomMatchTitle(room) }} · {{ $t(scheduleTime(room.schedule.starts_at)) }}{{ $t("（北京时间）") }}</h2>
          <p v-if="match?.finish_reason === 'late_forfeit' && match.winner_side === 'draw'">{{ $t("双方均未在宽限期内就位，本轮以 0:0 结束。") }}</p>
          <p v-else-if="match?.finish_reason === 'late_forfeit'">{{ $t("迟到判负：") }}{{ match.winner_side === 'yellow' ? room.schedule.yellow_name : room.schedule.white_name }}{{ (language==='zh'?' 获胜。':' wins.') }}</p>
          <template v-else><p>{{ $t("可提前签到与准备，到点后开始抽签。") }}{{ $t(scheduleTime(room.schedule.late_at)) }}{{ language==='zh'?` 后仍未全员就位的一方判 0:${roomGameKeys.length} 负；双方均未就位则 0:0。`:`: a team not ready forfeits 0:${roomGameKeys.length}; both absent means 0:0.` }}</p>
          <template v-if="room.schedule.players.length"><p v-for="side in ['yellow','white']" :key="side">{{ $t(room.schedule[`${side}_name`]) }}：{{ room.schedule.players.filter(p => p.side === side).map(p => `${p.position} 号 · ${p.display_name}（${p.arrived_at ? '已签到' : '未到场'}）`).join('、') }}</p></template>
          <p v-else>{{ $t("自由房间，已登录选手可自行落座。") }}</p>
          <p v-if="room.schedule.exception === 'both_late'" role="alert">{{ $t("双方均未在宽限期内就位，本轮以 0:0 结束。") }}</p></template>
        </section>
        <p v-if="error" class="alert">{{ $t(error) }}</p>
        <p v-if="room.replacement_room_code" class="alert">{{ $t("本房间已取消并安排重赛。") }}<a :href="`/rooms/${room.replacement_room_code}`">{{ $t("进入新房间 ") }}{{ $t(room.replacement_room_code) }}</a>{{ $t("，请重新落座。") }}</p>
        <p v-if="room.member_hold && !match?.suspension?.active" class="alert" role="status">{{ $t("参赛人员已被移出，流程暂缓。请房主或赛事管理员处理后继续。") }}</p>

        <Transition name="suspension-drop">
        <section v-if="match?.suspension?.active" class="suspension-banner" role="status">
          <div class="suspension-copy">
            <p class="eyebrow">MATCH SUSPENDED</p>
            <h2>{{ $t("比赛已由赛事方暂停") }}</h2>
            <p>{{ $t(match.suspension.reason_text) }}</p>
            <small>{{ match.suspension.started_by_display_name || `赛事方 #${match.suspension.started_by_user_id}` }}{{ $t(" · 两队计时器已冻结") }}</small>
          </div>
          <div class="resume-status">
            <span :class="match.suspension.resume_readiness.yellow.ready && 'ready'">{{ $t("黄方队长 ") }}{{ $t(match.suspension.resume_readiness.yellow.ready ? '已就绪' : '待就绪') }}</span>
            <span :class="match.suspension.resume_readiness.white.ready && 'ready'">{{ $t("白方队长 ") }}{{ $t(match.suspension.resume_readiness.white.ready ? '已就绪' : '待就绪') }}</span>
          </div>
          <div class="suspension-actions">
            <button v-if="room.me.can_mark_resume_ready" class="secondary-button" type="button" :disabled="busy" @click="toggleResumeReadiness">
              {{ $t(match.suspension.resume_readiness[mySeat.side].ready ? '取消恢复就绪' : '队长确认可恢复') }}
            </button>
            <button v-if="room.me.is_match_official" class="primary-button" type="button" :disabled="busy || !room.me.can_resume" @click="resumeMatch">{{ $t("恢复比赛与计时") }}</button>
          </div>
        </section>
        </Transition>

        <section v-if="room.me.can_manage_members || (match && room.status.startsWith('GAME_'))" class="match-operations">
          <details v-if="room.me.can_manage_members" class="operation-card"><summary>{{ $t("人员管理") }}</summary>
            <RoomMemberManagement v-model:user-id="manageUserId" :seats="room.seats" :removed-members="room.me.removed_members" :current-user-id="session?.user?.user_id" :busy="busy" @remove="manageMember($event)" @restore="manageMember($event, false)" />
          </details>
          <details v-if="room.me.can_report_issue || room.issues.length" class="operation-card" :open="issueOpen" @toggle="issueOpen = $event.target.open">
            <summary>{{ $t("问题上报 ") }}<span v-if="room.issues.length">{{ $t(room.issues.length) }}{{ $t(" 项待处理") }}</span></summary>
            <button type="button" class="secondary-button" @click="issueOpen = false">{{ $t("关闭上报面板 ×") }}</button>
            <p>{{ $t("提交后通知房间主办方／裁判处理，不会自动暂停。无人值守时请另行联系赛事管理员；收到暂停通知前，比赛仍继续计时。") }}</p>
            <div v-if="room.me.can_report_issue" class="operation-form issue-form">
              <select v-model="issueCategory" :aria-label='$t("问题类别")'>
                <option value="network_device">{{ $t("网络 / 设备") }}</option>
                <option value="project">{{ $t("项目运行") }}</option>
                <option value="rules">{{ $t("规则争议") }}</option>
                <option value="other">{{ $t("其他") }}</option>
              </select>
              <input v-model="issueDetails" maxlength="500" :placeholder='$t("简要描述问题（至少 3 个字符）")' />
              <button class="secondary-button" type="button" :disabled="busy || issueDetails.trim().length < 3" @click="reportIssue">{{ $t("提交给裁判") }}</button>
            </div>
            <div v-if="room.issues.length" class="issue-list">
              <article v-for="issue in room.issues" :key="issue.id">
                <div><strong>{{ issue.reporter_display_name }}</strong><span>{{ $t(issue.details) }}</span><small>{{ $t(issue.category) }}</small></div>
                <div v-if="room.me.is_match_official" class="issue-resolution">
                  <input v-model="issueResolutionNote" maxlength="500" :placeholder='$t("处理说明")' />
                  <button type="button" :disabled="busy || issueResolutionNote.trim().length < 3" @click="resolveIssue(issue)">{{ $t("解决") }}</button>
                  <button type="button" :disabled="busy || issueResolutionNote.trim().length < 3" @click="resolveIssue(issue, 'dismissed')">{{ $t("驳回") }}</button>
                </div>
              </article>
            </div>
          </details>

          <details v-if="match && room.me.is_match_official" class="operation-card referee-card">
            <summary>{{ $t("裁判控制台 ") }}<span>{{ $t("所有操作写入审计事件") }}</span></summary>
            <div class="referee-grid">
              <form v-if="room.me.can_suspend" class="operation-form" @submit.prevent="suspendMatch">
                <strong>{{ $t("暂停比赛") }}</strong>
                <select v-model="suspendReasonCode">
                  <option value="network_device">{{ $t("网络 / 设备") }}</option>
                  <option value="project">{{ $t("项目故障") }}</option>
                  <option value="rules">{{ $t("规则争议") }}</option>
                  <option value="medical">{{ $t("医疗情况") }}</option>
                  <option value="other">{{ $t("其他") }}</option>
                </select>
                <input v-model="suspendReasonText" maxlength="500" :placeholder='$t("公开暂停原因")' />
                <button class="danger-button" type="submit" :disabled="busy || suspendReasonText.trim().length < 3">{{ $t("暂停并冻结计时") }}</button>
              </form>

              <form v-if="room.me.can_override_result" class="operation-form result-correction" @submit.prevent="overrideResult">
                <strong>{{ $t("纠正当前赛果") }}</strong>
                <label>{{ $t("黄方分数 ") }}<input v-model.number="overrideYellowScore" type="number" min="0" /></label>
                <label>{{ $t("白方分数 ") }}<input v-model.number="overrideWhiteScore" type="number" min="0" /></label>
                <select v-model="overrideWinner">
                  <option value="yellow">{{ $t("黄方胜") }}</option><option value="white">{{ $t("白方胜") }}</option><option value="draw">{{ $t("平局") }}</option>
                </select>
                <input v-model="overrideReason" maxlength="500" :placeholder='$t("纠正原因")' />
                <button class="secondary-button" type="submit" :disabled="busy || overrideReason.trim().length < 3">{{ $t("发布修订并清空确认") }}</button>
              </form>

              <form v-if="room.me.can_force_advance" class="operation-form" @submit.prevent="forceAdvance">
                <strong>{{ $t("强制推进") }}</strong>
                <input v-model="forceAdvanceReason" maxlength="500" :placeholder='$t("跳过队长确认的原因")' />
                <button class="secondary-button" type="submit" :disabled="busy || stageWait > 0 || forceAdvanceReason.trim().length < 3">{{ $t(stageWait ? `休整 ${stageWait} 秒` : '进入下一项目') }}</button>
              </form>

              <form v-if="room.me.can_force_finish" class="operation-form" @submit.prevent="forceFinish">
                <strong>{{ $t("强制结束全场") }}</strong>
                <select v-model="forceFinishWinner"><option value="yellow">{{ $t("黄方胜") }}</option><option value="white">{{ $t("白方胜") }}</option><option value="draw">{{ $t("平局") }}</option></select>
                <input v-model="forceFinishReason" maxlength="500" :placeholder='$t("裁决原因")' />
                <button class="danger-button" type="submit" :disabled="busy || forceFinishReason.trim().length < 3">{{ $t("结束并冻结结果") }}</button>
              </form>
            </div>
          </details>
        </section>

        <Transition name="stage-note">
          <div v-if="visibleStageMotion" :key="visibleStageMotion.to" :class="['stage-motion-note', visibleStageMotion.kind]" role="status" aria-live="polite">
            <span class="stage-motion-mark">{{ $t(visibleStageMotion.kind === 'game-start' ? '▶' : visibleStageMotion.kind === 'match-finished' ? '✓' : '◆') }}</span>
            <div>
              <strong v-if="visibleStageMotion.kind === 'pick-lock'">{{ $t("项目与 BAN 已由服务器锁定") }}</strong>
              <strong v-else-if="visibleStageMotion.kind === 'lineup-reveal'">{{ $t("双方阵容已锁定 · 项目 A 开局检查") }}</strong>
              <strong v-else-if="visibleStageMotion.kind === 'game-start'">{{ $t("项目 ") }}{{ $t(visibleStageMotion.game) }}{{ $t(" 已开局") }}</strong>
              <strong v-else-if="visibleStageMotion.kind === 'game-result'">{{ $t("项目 ") }}{{ $t(visibleStageMotion.game) }}{{ $t(" 双方已完成 · 赛果公布") }}</strong>
              <strong v-else-if="visibleStageMotion.kind === 'match-finished'">{{ language==='zh'?'赛果已冻结 · 全场结算':'Results frozen · Match settlement' }}</strong>
              <strong v-else>{{ $t(statusText[room.status] || room.status) }}</strong>
              <small v-if="visibleStageMotion.kind === 'game-start'">{{ $t("队伍包干时间已按服务器开局时刻计时") }}</small>
              <small v-else-if="visibleStageMotion.kind === 'lineup-reveal'">{{ language==='zh'?'双方出战安排已公开':'Both lineups are revealed' }}</small>
              <small v-else-if="visibleStageMotion.kind === 'pick-lock'">{{ $t(draftSource(visibleStageMotion.to === 'SECOND_PICK_BAN' ? 'A' : 'B') || '服务器确认') }}</small>
              <small v-else>{{ $t("以服务器确认的比赛状态为准") }}</small>
            </div>
          </div>
        </Transition>

        <section v-if="isGameStage" class="game-hud" :aria-label='$t("当前比赛状态")'>
          <div class="game-hud-side yellow">
            <button class="game-hud-back" type="button" :aria-label='$t("返回比赛列表")' @click="navigate(competitionHomePath)">←</button>
            <strong class="game-hud-series" :aria-label="$t(`黄方局分 ${match.series_score.yellow}`)">{{ $t(match.series_score.yellow) }}</strong>
            <span class="game-hud-team">{{ $t("黄方") }}<small>{{ $t("包干时间") }}</small></span>
            <strong class="game-hud-clock">{{ $t(formatTeamClock(teamClockMs('yellow'))) }}</strong>
          </div>
          <div class="game-hud-center">
            <img v-if="roomProjectIcon(match.project_key)" class="game-hud-project-icon" :src="roomProjectIcon(match.project_key)" alt="" />
            <div class="game-hud-project-copy"><small class="game-hud-room">{{ room.name }} · {{ $t(room.room_code) }}</small><strong>{{ $t("项目 ") }}{{ $t(match.current_game_key) }} · {{ $t(projectName(match.project_key)) }}</strong></div>
            <div class="game-hud-progress" :aria-label="language==='zh'?'对局进度':'Game progress'"><span v-for="game in roomGameKeys" :key="game" :class="[game === currentGameKey && 'current', resultForGame(game) && 'complete']">{{ $t(game) }}</span></div>
          </div>
          <div class="game-hud-side white">
            <strong class="game-hud-clock">{{ $t(formatTeamClock(teamClockMs('white'))) }}</strong>
            <span class="game-hud-team">{{ $t("白方") }}<small>{{ $t("包干时间") }}</small></span>
            <strong class="game-hud-series" :aria-label="$t(`白方局分 ${match.series_score.white}`)">{{ $t(match.series_score.white) }}</strong>
            <span class="game-hud-stage"><i :class="['connection-dot', connection]"></i>{{ $t(match.suspension.active ? '暂停' : statusText[room.status]) }}</span>
          </div>
        </section>

        <div v-if="match && room.status.startsWith('GAME_') && !isGameStage" class="series-track" :aria-label="language==='zh'?'对局赛程':'Game schedule'">
          <div v-for="game in roomGameKeys" :key="game" :class="['series-game', game === currentGameKey && room.status !== 'FINISHED' && 'current', resultForGame(game) && 'complete']">
            <img v-if="roomProjectIcon(gameProject(game))" class="project-icon series-icon" :src="roomProjectIcon(gameProject(game))" alt="" />
            <span>{{ $t("项目 ") }}{{ $t(game) }}<small>{{ $t(resultForGame(game) ? `${projectResultValue(resultForGame(game), 'yellow')} / ${projectResultValue(resultForGame(game), 'white')} · ${winnerName(resultForGame(game).winner_side)}` : game === currentGameKey && room.status !== 'FINISHED' ? '当前项目' : '待进行') }}</small></span>
          </div>
        </div>

        <Transition :name="visibleStageMotion ? 'stage-swap' : ''">
        <div :key="room.status" class="phase-body">
        <section v-if="room.status === 'CANCELLED'" class="closed-room-panel">
          <p class="eyebrow">ROOM CLOSED</p>
          <h2>{{ $t("房间已关闭") }}</h2>
          <p>{{ $t("这场比赛未进入抽签，不能继续落座或开赛。房间记录已保留，可返回大厅查看。") }}</p>
          <button class="secondary-button" type="button" @click="navigate(competitionHomePath)">{{ $t("返回比赛大厅") }}</button>
        </section>
        <template v-else-if="isLobby">
          <section class="teams-grid">
            <div class="team-panel yellow-team">
              <div class="team-heading">
                <div><p class="eyebrow">YELLOW SIDE</p><h2>{{ room.schedule?.yellow_name || '黄队' }}</h2></div>
                <span :class="['ready-chip', room.teams.yellow.ready && 'is-ready']">{{ $t(room.teams.yellow.ready ? '已准备' : '未准备') }}</span>
              </div>
              <button
                v-for="item in teamSeats('yellow')"
                :key="item.position"
                :class="['seat-card', item.seat && 'occupied', item.seat?.user_id === room.me.user_id && 'mine']"
                type="button"
                :disabled="!canClaim(item.seat,item.side,item.position)"
                @click="claim(item.side, item.position)"
              >
                <PlayerAvatar v-if="item.seat" class="seat-avatar" :person="item.seat" />
                <span v-else class="seat-number">{{ $t(seatLabel(item.side, item.position)) }}</span>
                <span v-if="item.seat" class="seat-player"><strong>{{ item.seat.display_name }}</strong><small>{{ $t(item.position === 1 ? '队长席位' : '队员席位') }}</small></span>
                <span v-else class="seat-empty"><strong>{{ $t("空位") }}</strong><small>{{ $t(room.me.can_claim_seat ? '点击落座' : '等待选手') }}</small></span>
                <span v-if="item.seat?.user_id === room.me.user_id" class="you-label">{{ $t("你") }}</span>
              </button>
            </div>

            <div class="versus-column">
              <span class="versus">VS</span>
              <span v-if="room.schedule?.players.length">{{ room.schedule.players.filter(p => p.arrived_at).length }} / {{ roomPositions.length * 2 }} {{ language==='zh'?'已签到':'checked in' }}</span>
              <span v-else>{{ room.seats.length }} / {{ roomPositions.length * 2 }} {{ language==='zh'?'已落座':'seated' }}</span>
              <span v-if="room.status === 'SEATING'">{{ $t("等待全部选手") }}</span>
              <span v-else>{{ $t("等待双方队长") }}</span>
            </div>

            <div class="team-panel white-team">
              <div class="team-heading">
                <div><p class="eyebrow">WHITE SIDE</p><h2>{{ room.schedule?.white_name || '白队' }}</h2></div>
                <span :class="['ready-chip', room.teams.white.ready && 'is-ready']">{{ $t(room.teams.white.ready ? '已准备' : '未准备') }}</span>
              </div>
              <button
                v-for="item in teamSeats('white')"
                :key="item.position"
                :class="['seat-card', item.seat && 'occupied', item.seat?.user_id === room.me.user_id && 'mine']"
                type="button"
                :disabled="!canClaim(item.seat,item.side,item.position)"
                @click="claim(item.side, item.position)"
              >
                <PlayerAvatar v-if="item.seat" class="seat-avatar" :person="item.seat" />
                <span v-else class="seat-number">{{ $t(seatLabel(item.side, item.position)) }}</span>
                <span v-if="item.seat" class="seat-player"><strong>{{ item.seat.display_name }}</strong><small>{{ $t(item.position === 1 ? '队长席位' : '队员席位') }}</small></span>
                <span v-else class="seat-empty"><strong>{{ $t("空位") }}</strong><small>{{ $t(room.me.can_claim_seat ? '点击落座' : '等待选手') }}</small></span>
                <span v-if="item.seat?.user_id === room.me.user_id" class="you-label">{{ $t("你") }}</span>
              </button>
            </div>
          </section>

          <section class="action-bar">
            <div>
              <strong v-if="mySeat">{{ $t("你位于 ") }}{{ $t(seatLabel(mySeat.side, mySeat.position)) }}{{ $t(mySeat.position === 1 ? '，是本队队长' : '') }}</strong>
              <strong v-else-if="room.schedule">{{ $t("按报名表对应席位落座") }}</strong>
              <strong v-else>{{ $t("请选择一个空位落座") }}</strong>
              <p v-if="room.schedule">{{ $t("请按报名序号落座。队长准备前请核对双方名单；双方就位后等待预定开战时间开始抽签。") }}</p>
              <p v-else-if="room.status === 'SEATING'">{{ language==='zh'?'全部席位坐满后，双方队长可以准备。':'Once all seats are filled, both captains can ready up.' }}</p>
              <p v-else>{{ $t("任一队准备后席位将锁定，双方准备后进入抽签。") }}</p>
            </div>
            <div class="action-buttons">
              <button v-if="room.me.can_leave_seat" class="secondary-button" type="button" :disabled="busy" @click="leave">{{ $t("离开席位") }}</button>
              <button v-if="room.me.can_ready" class="primary-button" type="button" :disabled="busy" @click="toggleReady">{{ $t(busy ? '提交中…' : myTeamReady ? '取消准备' : '队长准备') }}</button>
            </div>
          </section>
        </template>

        <section v-else-if="room.status === 'DRAW'" :class="['draw-stage', visibleStageMotion?.kind === 'first-draw' && revealStep === 0 && 'drawing']">
          <p class="eyebrow">FIRST SIDE DRAW</p>
          <div class="draw-orbit"><span v-if="visibleStageMotion?.kind === 'first-draw' && revealStep === 0" class="draw-alternating"><b>{{ $t("黄") }}</b><b>{{ $t("白") }}</b></span><span v-else>{{ $t(sideName(draft.first_side).slice(0, 1)) }}</span></div>
          <h1>{{ $t(visibleStageMotion?.kind === 'first-draw' && revealStep === 0 ? '抽签结果揭晓中' : `${sideName(draft.first_side)}获得先手`) }}</h1>
          <p>{{ language==='zh'?`${captainName(draft.first_side)} 首步选择 ${room.rules.steps[0].picks} 项、禁用 ${room.rules.steps[0].bans} 项。`:`${captainName(draft.first_side)} opens with ${room.rules.steps[0].picks} pick(s) and ${room.rules.steps[0].bans} ban(s).` }}</p>
          <strong class="draw-countdown">{{ $t(formatCountdown(remainingSeconds)) }}</strong>
          <small>{{ $t("抽签承诺 ") }}{{ $t(draft.commitment.slice(0, 12)) }}… · {{ $t(draft.algorithm_version) }}</small>
        </section>

        <DraftWorkflow v-else-if="draft?.workflow && ['DRAFT_STEP','BLIND_PICK','C_DRAW'].includes(room.status)" :workflow="draft.workflow" :projects="room.projects" :phase="room.status" :token="draft.phase_token" :lang="language" :can-submit="room.me.can_submit_pick_ban" :can-blind="room.me.can_submit_blind" :blind-submitted="!!draft.my_blind_choice" :busy="busy" :seconds="remainingSeconds" :name-of="projectName" :icon-of="roomProjectIcon" @submit="values => perform(() => api.submitDraftStep(currentCode,values,draft.phase_token))" @blind="key => perform(() => api.submitBlind(currentCode,key,draft.phase_token))" />

        <template v-else-if="['FIRST_PICK_BAN', 'SECOND_PICK_BAN', 'BLIND_PICK'].includes(room.status)">
          <section class="bp-team-bar">
            <div :class="['bp-team-side', 'yellow', draft.active_side === 'yellow' && 'is-active']">
              <span>{{ $t("黄方") }}</span><strong>{{ $t(captainName('yellow')) }}</strong>
              <small>{{ $t(draft.first_side === 'yellow' ? '先手' : '后手') }}</small>
            </div>
            <div class="bp-phase-title">
              <span>{{ $t(statusText[room.status]) }}</span>
              <strong>{{ $t(formatCountdown(remainingSeconds)) }}</strong>
              <small v-if="room.status !== 'BLIND_PICK'">{{ $t(sideName(draft.active_side)) }} · {{ $t("队长操作") }}</small>
              <small v-else>{{ $t("双方独立提交，选择互不可见") }}</small>
            </div>
            <div :class="['bp-team-side', 'white', draft.active_side === 'white' && 'is-active']">
              <span>{{ $t("白方") }}</span><strong>{{ $t(captainName('white')) }}</strong>
              <small>{{ $t(draft.first_side === 'white' ? '先手' : '后手') }}</small>
            </div>
          </section>

          <section :class="['draft-history', visibleStageMotion?.kind === 'pick-lock' && 'locking']">
            <div data-slot-key="A" :class="visibleStageMotion?.to === 'SECOND_PICK_BAN' && 'new-lock'"><img v-if="roomProjectIcon(draft.project_a)" class="project-icon history-icon" :src="roomProjectIcon(draft.project_a)" alt="" /><span>{{ $t("项目 A ") }}<em>{{ $t(draftSource('A')) }}</em></span><strong>{{ $t(projectName(draft.project_a)) }}</strong></div>
            <div data-slot-key="M" :class="['ban', visibleStageMotion?.to === 'SECOND_PICK_BAN' && 'new-lock']"><img v-if="roomProjectIcon(draft.ban_m)" class="project-icon history-icon" :src="roomProjectIcon(draft.ban_m)" alt="" /><span>{{ $t(sideName(draft.first_side)) }} BAN <em>{{ $t(draftSource('A')) }}</em></span><strong>{{ $t(projectName(draft.ban_m)) }}</strong></div>
            <div data-slot-key="B" :class="visibleStageMotion?.to === 'BLIND_PICK' && 'new-lock'"><img v-if="roomProjectIcon(draft.project_b)" class="project-icon history-icon" :src="roomProjectIcon(draft.project_b)" alt="" /><span>{{ $t("项目 B ") }}<em>{{ $t(draftSource('B')) }}</em></span><strong>{{ $t(projectName(draft.project_b)) }}</strong></div>
            <div data-slot-key="N" :class="['ban', visibleStageMotion?.to === 'BLIND_PICK' && 'new-lock']"><img v-if="roomProjectIcon(draft.ban_n)" class="project-icon history-icon" :src="roomProjectIcon(draft.ban_n)" alt="" /><span>{{ $t(sideName(draft.second_side)) }} BAN <em>{{ $t(draftSource('B')) }}</em></span><strong>{{ $t(projectName(draft.ban_n)) }}</strong></div>
          </section>

          <section v-if="room.status !== 'BLIND_PICK'" class="bp-workspace">
            <div class="bp-instruction">
              <div>
                <p class="eyebrow">PICK + BAN</p>
                <h2 v-if="room.me.can_submit_pick_ban">{{ $t("选择一个比赛项目，并 BAN 一个不同项目") }}</h2>
                <h2 v-else>{{ $t("等待 ") }}{{ $t(sideName(draft.active_side)) }}{{ $t("队长完成选择") }}</h2>
              </div>
              <div v-if="room.me.can_submit_pick_ban" class="current-choices">
                <span>PICK <strong>{{ $t(projectName(selectedPick)) }}</strong></span>
                <span>BAN <strong>{{ $t(projectName(selectedBan)) }}</strong></span>
              </div>
            </div>
            <div class="project-pool">
              <article
                v-for="project in room.projects"
                :key="project.key"
                :data-pool-key="project.key"
                :class="['project-card', !isProjectAvailable(project.key) && 'unavailable', selectedPick === project.key && 'selected-pick', selectedBan === project.key && 'selected-ban']"
              >
                <span class="project-order">{{ $t(String(project.sort_order).padStart(2, '0')) }}</span>
                <img v-if="projectIconUrl(project.project_ref)" class="project-icon pool-icon" :src="projectIconUrl(project.project_ref)" alt="" />
                <h3>{{ $t(project.name) }}</h3>
                <p>{{ $t(project.description || `规则版本 ${project.rules_version}`) }}</p>
                <div v-if="room.me.can_submit_pick_ban && isProjectAvailable(project.key)" class="project-actions">
                  <button type="button" :class="selectedPick === project.key && 'active'" @click="choosePick(project.key)">{{ $t("选择") }}</button>
                  <button type="button" :class="['ban-action', selectedBan === project.key && 'active']" @click="chooseBan(project.key)">BAN</button>
                </div>
                <span v-else-if="draft.project_a === project.key" class="project-result">{{ $t("项目 A") }}</span>
                <span v-else-if="draft.project_b === project.key" class="project-result">{{ $t("项目 B") }}</span>
                <span v-else-if="[draft.ban_m, draft.ban_n].includes(project.key)" class="project-result banned">{{ $t(projectMark(project.key)) }}</span>
              </article>
            </div>
            <div v-if="room.me.can_submit_pick_ban" class="draft-submit-bar">
              <span>{{ $t("超时后系统按项目池顺序选择首个合法组合。") }}</span>
              <button class="primary-button" type="button" :disabled="busy || !canSubmitPickBan" @click="submitPickBan">{{ $t("锁定并公开") }}</button>
            </div>
          </section>

          <section v-else class="bp-workspace blind-workspace">
            <div class="bp-instruction">
              <div><p class="eyebrow">SEALED PICK</p><h2>{{ $t("双方同时盲选项目 C 候选") }}</h2></div>
              <div class="blind-status">
                <span :class="draft.blind_submissions.yellow && 'submitted'">{{ $t("黄方 ") }}{{ $t(draft.blind_submissions.yellow ? '已提交' : '选择中') }}</span>
                <span :class="draft.blind_submissions.white && 'submitted'">{{ $t("白方 ") }}{{ $t(draft.blind_submissions.white ? '已提交' : '选择中') }}</span>
              </div>
            </div>
            <p v-if="ownRemainingSeconds !== null && room.me.is_captain" class="own-deadline">{{ $t("我的剩余时间：") }}{{ $t(formatCountdown(ownRemainingSeconds)) }}</p>
            <div class="blind-envelopes" :aria-label='$t("双方密封提交状态")'>
              <div v-for="side in ['yellow', 'white']" :key="side" :class="['blind-envelope', side, draft.blind_submissions[side] && 'sealed']">
                <span>{{ $t(sideName(side)) }}{{ $t("候选") }}</span>
                <strong>{{ $t(draft.blind_submissions[side] ? (mySeat?.side === side && draft.my_blind_choice ? projectName(draft.my_blind_choice) : '已密封') : '选择中') }}</strong>
                <small>{{ $t(draft.blind_submissions[side] ? '提交已锁定' : '尚未提交') }}</small>
              </div>
            </div>
            <div class="project-pool blind-pool">
              <button
                v-for="project in room.projects"
                :key="project.key"
                type="button"
                :class="['project-card', !isProjectAvailable(project.key) && 'unavailable', selectedBlind === project.key && 'selected-pick']"
                :disabled="!room.me.can_submit_blind || !isProjectAvailable(project.key)"
                @click="selectedBlind = project.key"
              >
                <span class="project-order">{{ $t(String(project.sort_order).padStart(2, '0')) }}</span>
                <img v-if="projectIconUrl(project.project_ref)" class="project-icon pool-icon" :src="projectIconUrl(project.project_ref)" alt="" />
                <h3>{{ $t(project.name) }}</h3>
                <p>{{ $t(project.description || `规则版本 ${project.rules_version}`) }}</p>
                <span v-if="projectMark(project.key)" class="project-result">{{ $t(projectMark(project.key)) }}</span>
              </button>
            </div>
            <div class="draft-submit-bar">
              <span v-if="draft.my_blind_choice">{{ $t("本队已密封提交：") }}{{ $t(projectName(draft.my_blind_choice)) }}</span>
              <span v-else-if="!room.me.can_submit_blind">{{ $t("等待双方队长提交；候选项目不会提前公开。") }}</span>
              <span v-else>{{ $t("选定后不可修改；超时使用第一个合法项目。") }}</span>
              <button v-if="room.me.can_submit_blind" class="primary-button" type="button" :disabled="busy || !canSubmitBlind" @click="submitBlind">{{ $t("密封提交") }}</button>
            </div>
          </section>
        </template>

        <section v-else-if="room.status === 'C_DRAW'" :class="['draft-complete', visibleStageMotion?.kind === 'c-draw' && revealStep < 2 && 'revealing', `reveal-step-${revealStep}`]">
          <p class="eyebrow">DRAFT COMPLETE</p>
          <h1>{{ $t(draft.c_reveal_at ? '双方盲选候选已公开' : '三个比赛项目已确定') }}</h1>
          <div v-if="draft.c_reveal_at" class="final-projects blind-candidate-cards">
            <div v-for="side in ['yellow', 'white']" :key="side"><img v-if="roomProjectIcon(draft.blind_choices[side])" class="project-icon summary-icon" :src="roomProjectIcon(draft.blind_choices[side])" alt="" /><span>{{ $t(sideName(side)) }}{{ $t("候选") }}</span><strong>{{ $t(projectName(draft.blind_choices[side])) }}</strong><small>{{ $t(projectDescription(draft.blind_choices[side])) }}</small></div>
          </div>
          <div v-else class="final-projects">
            <div><img v-if="roomProjectIcon(draft.project_a)" class="project-icon summary-icon" :src="roomProjectIcon(draft.project_a)" alt="" /><span>A · {{ $t(sideName(draft.first_side)) }}{{ $t("选择") }}</span><strong>{{ $t(projectName(draft.project_a)) }}</strong><small>{{ $t(projectDescription(draft.project_a)) }}</small></div>
            <div><img v-if="roomProjectIcon(draft.project_b)" class="project-icon summary-icon" :src="roomProjectIcon(draft.project_b)" alt="" /><span>B · {{ $t(sideName(draft.second_side)) }}{{ $t("选择") }}</span><strong>{{ $t(projectName(draft.project_b)) }}</strong><small>{{ $t(projectDescription(draft.project_b)) }}</small></div>
            <div class="project-c"><img v-if="roomProjectIcon(draft.project_c)" class="project-icon summary-icon" :src="roomProjectIcon(draft.project_c)" alt="" /><span>{{ $t("C · 盲选抽签") }}</span><strong>{{ $t(projectName(draft.project_c)) }}</strong><small>{{ $t(projectDescription(draft.project_c)) }}</small></div>
          </div>
          <div class="blind-reveal">
            <span v-if="draft.blind_choices.yellow === draft.blind_choices.white" class="blind-candidate">{{ $t("双方相同候选：") }}<strong>{{ $t(projectName(draft.blind_choices.yellow)) }}</strong> <em>{{ $t("黄 ") }}{{ $t(draftSource('yellow')) }}{{ $t(" · 白 ") }}{{ $t(draftSource('white')) }}</em></span>
            <template v-else>
              <span class="blind-candidate">{{ $t("黄方候选：") }}<strong>{{ $t(projectName(draft.blind_choices.yellow)) }}</strong> <em>{{ $t(draftSource('yellow')) }}</em></span>
              <span class="blind-candidate">{{ $t("白方候选：") }}<strong>{{ $t(projectName(draft.blind_choices.white)) }}</strong> <em>{{ $t(draftSource('white')) }}</em></span>
            </template>
            <span>BAN：{{ $t(projectName(draft.ban_m)) }} / {{ $t(projectName(draft.ban_n)) }}</span>
          </div>
          <p v-if="draft.c_reveal_at">{{ $t("先展示双方候选，随后揭晓项目 C 的抽签结果。") }}</p>
          <p v-else>{{ $t("抽签结果展示中，") }}{{ $t(formatCountdown(remainingSeconds)) }}{{ $t(" 后进入双方队长秘密布阵。") }}</p>
        </section>

        <template v-else-if="room.status === 'LINEUP'">
          <section class="bp-team-bar lineup-team-bar">
            <div class="bp-team-side yellow">
              <span>{{ $t("黄方") }}</span><strong>{{ $t(captainName('yellow')) }}</strong>
              <small :class="['submission-state', lineup.submissions.yellow && 'submitted']">{{ $t(lineup.submissions.yellow ? '阵容已密封' : '队长布阵中') }}</small>
            </div>
            <div class="bp-phase-title">
              <span>SECRET LINEUP</span>
              <strong>{{ $t(formatCountdown(remainingSeconds)) }}</strong>
              <small>{{ $t("双方独立计时 · ") }}{{ $t(ownRemainingSeconds === null ? '完整编排仅本队队员可见' : `我的剩余 ${formatCountdown(ownRemainingSeconds)}`) }}</small>
            </div>
            <div class="bp-team-side white">
              <span>{{ $t("白方") }}</span><strong>{{ $t(captainName('white')) }}</strong>
              <small :class="['submission-state', lineup.submissions.white && 'submitted']">{{ $t(lineup.submissions.white ? '阵容已密封' : '队长布阵中') }}</small>
            </div>
          </section>

          <section class="lineup-projects">
            <div v-for="game in roomGameKeys" :key="game">
              <img v-if="roomProjectIcon(gameProject(game))" class="project-icon lineup-icon" :src="roomProjectIcon(gameProject(game))" alt="" /><span>{{ $t("第 ") }}{{ $t(game) }}{{ $t(" 场") }}</span><strong>{{ $t(projectName(gameProject(game))) }}</strong>
            </div>
          </section>

          <section class="lineup-workspace">
            <div class="lineup-roster yellow-roster">
              <p class="eyebrow">YELLOW TEAM</p>
              <h2>{{ $t("黄方名单") }}</h2>
              <div v-for="item in teamSeats('yellow')" :key="item.position" class="lineup-player">
                <PlayerAvatar :person="item.seat" /><strong>{{ item.seat?.display_name }}</strong><small>{{ $t(item.position === 1 ? '队长' : '队员') }}</small>
              </div>
            </div>

            <div class="lineup-console">
              <template v-if="room.me.can_submit_lineup">
                <div class="lineup-title"><p class="eyebrow">CAPTAIN ONLY</p><h2>{{ language==='zh'?'安排各局出战选手':'Assign players to games' }}</h2><p>{{ lineupPolicyDescription(room.rules?.lineup_policy, language) }} {{ language==='zh'?'提交后不可修改。':'Lineups cannot be changed after submission.' }}</p><p v-if="room.rules?.series_mode==='best_of'">{{ language==='zh'?'限制按完整布阵校验，提前结束可能使部分选手未实际出场。':'Policies apply to the full planned lineup; an early finish may leave some players without an actual appearance.' }}</p></div>
                <label v-for="game in roomGameKeys" :key="game" class="lineup-assignment">
                  <span><b>{{ $t(game) }}</b><img v-if="roomProjectIcon(gameProject(game))" class="project-icon assignment-icon" :src="roomProjectIcon(gameProject(game))" alt="" /><small>{{ $t(projectName(gameProject(game))) }}</small></span>
                  <select v-model.number="lineupSelections[game]">
                    <option v-for="position in roomPositions" :key="position" :value="position">{{ $t(position) }}{{ $t(" 号位 · ") }}{{ $t(playerAt(mySeat.side, position)) }}</option>
                  </select>
                </label>
                <p v-if="!canSubmitLineup" class="lineup-warning">{{ language==='zh'?'当前阵容不符合本房间出场限制。':'This lineup does not satisfy the room’s appearance policy.' }}</p>
                <button class="primary-button lineup-submit" type="button" :disabled="busy || !canSubmitLineup" @click="submitLineup">{{ $t("密封提交阵容") }}</button>
                <small class="lineup-timeout">{{ language==='zh'?'超时按席位顺序循环安排出场。':'On timeout, players are assigned cyclically in seat order.' }}</small>
              </template>
              <template v-else-if="lineup.my_lineup">
                <div class="sealed-mark">✓</div>
                <h2>{{ $t("本队阵容已密封") }}</h2>
                <p>{{ language==='zh'?'完整编排仅本队队员可见。':'The sealed lineup is visible only to your team.' }}</p>
                <div v-for="game in roomGameKeys" :key="game" class="sealed-row">
                  <b>{{ $t(game) }}</b><span>{{ $t(projectName(gameProject(game))) }}</span><strong>{{ lineup.my_lineup[game].display_name }}</strong>
                </div>
              </template>
              <template v-else>
                <div class="sealed-mark waiting">•••</div>
                <h2>{{ $t(room.me.is_captain ? '等待对方队长提交' : '队长正在秘密布阵') }}</h2>
                <p>{{ language==='zh'?'提交前互相保密；双方都提交后，一并公开全部出战安排。':'Lineups are sealed until both captains submit, then revealed together.' }}</p>
              </template>
            </div>

            <div class="lineup-roster white-roster">
              <p class="eyebrow">WHITE TEAM</p>
              <h2>{{ $t("白方名单") }}</h2>
              <div v-for="item in teamSeats('white')" :key="item.position" class="lineup-player">
                <PlayerAvatar :person="item.seat" /><strong>{{ item.seat?.display_name }}</strong><small>{{ $t(item.position === 1 ? '队长' : '队员') }}</small>
              </div>
            </div>
          </section>
        </template>

        <template v-else-if="isGameReady">
          <p v-if="stageWait > 0" class="muted">{{ $t("开局展示剩余 ") }}{{ $t(stageWait) }}{{ $t(" 秒，可提前就绪；展示结束且双方就绪后开始，不扣比赛用时。") }}</p>
          <div v-if="lineup?.revealed_lineups" class="public-matchups"><p v-for="game in roomGameKeys" :key="game">{{ lineup.revealed_lineups.yellow[game]?.display_name }}{{ $t(" — 项目 ") }}{{ $t(game) }} · {{ $t(projectName(gameProject(game))) }} — {{ lineup.revealed_lineups.white[game]?.display_name }}</p></div>
          <p v-if="predictionWait > 0" class="muted" role="status">{{ $t("赛事下注最短窗口剩余 ") }}{{ $t(predictionWait) }}{{ $t(" 秒；双方就绪后将自动开局，此处等待不扣队伍用时。") }}</p>
          <section class="pregame-panel">
            <p>{{ $t("双方独立确认，剩余 ") }}{{ $t(Math.max(0,Math.ceil((Date.parse(match.ready_deadline_at)-clockNow)/1000)) || 0) }}{{ $t(" 秒后自动确认。期间不扣队伍包干时间。") }}</p>
            <div class="pregame-heading"><p class="eyebrow">PRE-GAME CHECK</p><h1>{{ $t("项目 ") }}{{ $t(match.current_game_key) }}{{ $t(" 开局检查") }}</h1><p>{{ $t("双方出战者和队长全部就绪后，项目与两队包干计时自动开始。") }}</p></div>
            <div class="pregame-project"><img v-if="roomProjectIcon(match.project_key)" class="project-icon pregame-project-icon" :src="roomProjectIcon(match.project_key)" alt="" /><div><span>{{ $t("本场项目") }}</span><h2>{{ $t(projectName(match.project_key)) }}</h2><p>{{ $t(projectDescription(match.project_key)) }}</p></div></div>
            <div class="pregame-versus">
              <div v-for="side in ['yellow', 'white']" :key="side" :class="['pregame-team', side]">
                <span class="side-kicker">{{ $t(sideName(side)) }}</span>
                <h2 class="pregame-player"><PlayerAvatar v-if="match.players[side]" :person="match.players[side]" />{{ match.players[side]?.display_name || '名单同步中' }}</h2>
                <small v-if="match.players[side]">{{ $t(match.players[side].position) }}{{ $t(" 号位 · 本场出战") }}</small>
                <small v-else>{{ $t("本队队员可查看本队安排") }}</small>
                <div class="check-row"><span>{{ $t("出战者连接与就绪") }}</span><b :class="match.readiness[side].player_ready && 'ready'">{{ $t(match.readiness[side].player_ready ? '已就绪' : '未就绪') }}</b></div>
                <div class="check-row"><span>{{ $t("队长确认") }}</span><b :class="match.readiness[side].captain_ready && 'ready'">{{ $t(match.readiness[side].captain_ready ? '已确认' : '未确认') }}</b></div>
              </div>
            </div>
            <div class="pregame-actions">
              <button v-if="room.me.can_mark_player_ready" class="secondary-button" type="button" :disabled="busy" @click="toggleGameReadiness('player')">{{ $t(match.readiness[mySeat.side].player_ready ? '取消出战者就绪' : '我是出战者，已就绪') }}</button>
              <button v-if="room.me.can_mark_captain_ready" class="secondary-button" type="button" :disabled="busy" @click="toggleGameReadiness('captain')">{{ $t(match.readiness[mySeat.side].captain_ready ? '取消队长确认' : '队长确认开局') }}</button>
              <span v-if="!room.me.can_mark_player_ready && !room.me.can_mark_captain_ready">{{ $t("等待双方出战者与队长完成开局检查。") }}</span>
            </div>
          </section>
        </template>

        <template v-else-if="showPlayingBoards">
          <section class="game-play-layout">
            <div class="game-roster-edge yellow" :aria-label='$t("黄方队员状态")'>
              <div class="roster-edge-title">{{ $t("黄方队员") }}</div>
              <div v-for="item in teamSeats('yellow')" :key="item.position" :class="['roster-edge-player', Number(match.players.yellow?.position) === item.position && 'active', match.sessions.yellow?.finished && Number(match.players.yellow?.position) === item.position && 'complete']">
                <PlayerAvatar class="roster-edge-avatar" :person="item.seat" />
                <span class="roster-edge-copy"><strong>{{ item.seat?.display_name || `黄${item.position}` }}</strong><small>{{ $t(item.position === 1 ? '队长 · ' : '') }}{{ $t(rosterStatus('yellow', item.position)) }}</small></span>
              </div>
            </div>
            <div class="project-dual-view">
              <article v-for="side in ['yellow', 'white']" :key="side" :class="['project-side-view', side, match.sessions[side]?.finished && 'finished']">
                <header class="project-metrics" :style="{'--metric-count':ruleMetrics(side).length+2}" :aria-label="$t(`${sideName(side)}本场数据`)">
                  <div class="project-metric performance"><span>{{ $t(projectMetric(side).label) }}</span><strong>{{ $t(projectMetric(side).value) }}</strong></div>
                  <div v-for="item in ruleMetrics(side)" :key="item.key" :class="['project-metric', 'rule-metric', {warning:item.warning}]"><span>{{ item.label }}</span><strong>{{ item.value }}</strong></div>
                  <div class="project-metric time"><span>{{ $t(projectRemainingMs(side) == null ? '用时' : '倒计时') }}</span><strong>{{ $t(formatProjectElapsed(projectRemainingMs(side) ?? projectElapsedMs(side))) }}</strong></div>
                </header>
                <div class="project-board-stage" :class="match.sessions[side]?.public_view?.view_protocol === 'cargo-transport-v1' && 'cargo-board-stage'">
                <div v-if="match.sessions[side]?.finished" class="board-complete-tag" role="status">{{ $t("本侧已完成") }}</div>
                <div v-if="sessionPayload(side)?.awaiting_client" class="project-view-fallback" role="status">{{ $t("等待选手载入棋盘…") }}</div>
                <ObservedProjectBoard v-else-if="!isMyActiveSide(side)" class="embedded-project-board"
                  :view="match.sessions[side]?.public_view" :stream-key="match.sessions[side]?.instance_id"
                  @gap="disconnectRoom?.resync?.()"
                  @pending="setObserverPending(side, $event)" />
                <CargoBoard
                  v-else-if="match.sessions[side]?.public_view?.view_protocol === 'cargo-transport-v1'"
                  class="embedded-project-board"
                  :snapshot="projectBoardSnapshot(side)"
                  :disabled="!canUseBoard(side)"
                  :aria-label="$t(`${sideName(side)}真·华容道棋盘`)"
                  @move="moveGame"
                />
                <TournamentBoard
                  v-else-if="['2048-board-v1', '2048-board-v2'].includes(match.sessions[side]?.public_view?.view_protocol)"
                  class="embedded-project-board"
                  :snapshot="projectBoardSnapshot(side)"
                  :mirror-portals="Boolean(sessionPayload(side)?.mirror_portals)"
                  :irregular-shape="Boolean(sessionPayload(side)?.shape_shifter || sessionPayload(side)?.aftershock)"
                  :aftershock="Boolean(sessionPayload(side)?.aftershock)"
                  :show-dice-effect="Boolean(sessionPayload(side)?.dice)"
                  :sealed-cells="sessionPayload(side)?.sealed_cells || []"
                  :disabled="!canUseBoard(side)"
                  :aria-label="$t(`${sideName(side)}项目棋盘`)"
                  @move="moveGame"
                />
                <PolyominoBoard
                  v-else-if="match.sessions[side]?.public_view?.view_protocol === 'polyomino-board-v1'"
                  class="embedded-project-board"
                  :snapshot="projectBoardSnapshot(side)"
                  :disabled="!canUseBoard(side)"
                  :aria-label="$t(`${sideName(side)}越来越大棋盘`)"
                  @move="moveGame"
                />
                <div v-else class="project-view-fallback"><strong>{{ $t("项目公开画面暂不可用") }}</strong><span>{{ $t(match.sessions[side]?.public_view?.view_kind || '等待项目状态') }}</span></div>
                <div v-if="isMyActiveSide(side) && projectThinking" class="project-thinking-pill" role="status"><span></span>{{ $t("AI 思考中") }}</div>
                </div>
                <p v-if="isMyActiveSide(side) && sessionPayload(side)?.no_moves && (sessionPayload(side)?.allow_undo || sessionPayload(side)?.allow_restart)" class="project-recovery-note">{{ $t("当前盘面无可用移动，") }}{{ $t(sessionPayload(side)?.allow_undo ? '撤销' : '重开') }}{{ $t("后可继续。") }}</p>
                <button v-if="isMyActiveSide(side) && canSurrender" class="secondary-button" type="button" :disabled="movePending" @click="projectAction('surrender')">{{ $t("认输本局（保留当前得分）") }}</button>
                <template v-if="isMyActiveSide(side)">
                  <div v-if="room.me.can_move" class="move-pad side-move-pad" :aria-label='$t("棋盘方向操作")'>
                    <button type="button" :aria-label='$t("向上")' :disabled="!canUseBoard(side)" @click="moveGame('up')">↑</button>
                    <button type="button" :aria-label='$t("向左")' :disabled="!canUseBoard(side)" @click="moveGame('left')">←</button>
                    <button type="button" :aria-label='$t("向下")' :disabled="!canUseBoard(side)" @click="moveGame('down')">↓</button>
                    <button type="button" :aria-label='$t("向右")' :disabled="!canUseBoard(side)" @click="moveGame('right')">→</button>
                  </div>
                  <div v-if="room.me.can_move && (sessionPayload(side)?.allow_undo || sessionPayload(side)?.allow_restart)" class="project-action-row">
                    <button v-if="sessionPayload(side)?.allow_undo" v-touch-click type="button" :disabled="!canUseBoard(side) || !sessionPayload(side)?.can_undo" @click="projectAction('undo')">{{ $t("撤销一步（Z）") }}</button>
                    <button v-if="sessionPayload(side)?.allow_restart" type="button" :disabled="!canUseBoard(side)" @click="projectAction('restart')">{{ $t("重新开始（R）") }}</button>
                  </div>
                  <p v-if="room.me.can_move" class="move-help">{{ $t("方向键 / WASD 操作 · 对手棋盘可实时查看") }}</p>
                  <p v-if="localPacket && projectSyncState === 'reconnecting'" class="project-recovery-note" role="status">{{ $t("网络重连中，当前进度已在本机保留，可继续操作。") }}</p>
                  <p v-else-if="localPacket?.finished && projectSyncState === 'syncing'" class="project-recovery-note" role="status">{{ $t("本局已完成，正在同步成绩…") }}</p>
                  <div v-else-if="match.suspension.active" class="project-finished-note suspended-note"><strong>{{ $t("比赛暂停") }}</strong><span>{{ $t("操作已锁定，等待双方队长与裁判恢复比赛。") }}</span></div>
                </template>
              </article>
            </div>
            <div class="game-roster-edge white" :aria-label='$t("白方队员状态")'>
              <div class="roster-edge-title">{{ $t("白方队员") }}</div>
              <div v-for="item in teamSeats('white')" :key="item.position" :class="['roster-edge-player', Number(match.players.white?.position) === item.position && 'active', match.sessions.white?.finished && Number(match.players.white?.position) === item.position && 'complete']">
                <span class="roster-edge-copy"><strong>{{ item.seat?.display_name || `白${item.position}` }}</strong><small>{{ $t(item.position === 1 ? '队长 · ' : '') }}{{ $t(rosterStatus('white', item.position)) }}</small></span>
                <PlayerAvatar class="roster-edge-avatar" :person="item.seat" />
              </div>
            </div>
            <footer class="game-rules-footer"><span>{{ $t("当前玩法 · 项目 ") }}{{ $t(match.current_game_key) }}</span><strong>{{ $t(projectName(match.project_key)) }}</strong><p>{{ $t(projectDescription(match.project_key)) }}</p><LanguageSwitch /></footer>
          </section>
        </template>

        <section v-else-if="isGameResult" class="game-result-panel">
          <div class="result-versus">
            <div class="result-side yellow"><div class="result-player"><PlayerAvatar :person="match.players.yellow" /><span>{{ $t("黄方 · ") }}{{ match.players.yellow.display_name }}</span></div><strong>{{ $t(projectResultValue(match.current_result, 'yellow')) }}</strong></div>
            <div class="result-center"><img v-if="roomProjectIcon(match.project_key)" class="project-icon result-icon" :src="roomProjectIcon(match.project_key)" alt="" /><span>{{ $t("项目 ") }}{{ $t(match.current_game_key) }} · {{ $t(projectName(match.project_key)) }}</span><b>{{ $t(match.current_result.winner_side === 'draw' ? '平局' : `${sideName(match.current_result.winner_side)}获胜`) }}</b><p>{{ $t(projectDescription(match.project_key)) }}</p></div>
            <div class="result-side white"><div class="result-player"><PlayerAvatar :person="match.players.white" /><span>{{ $t("白方 · ") }}{{ match.players.white.display_name }}</span></div><strong>{{ $t(projectResultValue(match.current_result, 'white')) }}</strong></div>
          </div>
          <p v-if="match.current_result.corrected" class="result-correction-note">{{ $t("裁判已修订 · ") }}{{ $t(match.current_result.correction_reason) }}</p>
          <p v-if="match.current_result.reason?.endsWith('_surrendered')">{{ $t(sideName(match.current_result.reason.split('_')[0])) }}{{ $t("认输本局，保留认输时成绩并判负。") }}</p>
          <div class="result-boards"><div v-for="side in ['yellow', 'white']" :key="side"><strong>{{ $t(sideName(side)) }}{{ $t("最终盘面") }}</strong>
            <component :is="match.sessions[side]?.public_view?.view_protocol === 'cargo-transport-v1' ? CargoBoard : match.sessions[side]?.public_view?.view_protocol === 'polyomino-board-v1' ? PolyominoBoard : TournamentBoard" v-if="sessionPayload(side)?.board" :snapshot="projectBoardSnapshot(side)" :disabled="true" :mirror-portals="Boolean(sessionPayload(side)?.mirror_portals)" :irregular-shape="Boolean(sessionPayload(side)?.shape_shifter || sessionPayload(side)?.aftershock)" :aftershock="Boolean(sessionPayload(side)?.aftershock)" :sealed-cells="sessionPayload(side)?.sealed_cells || []" />
          </div></div>
          <p v-for="side in ['yellow','white']" :key="`refund-${side}`" v-show="match.current_result[`${side}_refund_ms`]>0">{{ $t(sideName(side)) }}{{ $t("包干补时 +") }}{{ $t((match.current_result[`${side}_refund_ms`]/1000).toFixed(2)) }}{{ $t(" 秒") }}</p>
          <p v-if="stageWait">{{ $t("休整剩余 ") }}{{ $t(stageWait) }}{{ $t(" 秒，随后自动继续。") }}</p>
          <div class="confirmation-strip">
            <strong>{{ $t("休整结束后自动进入") }}{{ $t(match.current_game_key==='C' ? '全场结算' : '下一项目') }}</strong>
          </div>
        </section>

        <MatchSettlement v-else-if="room.status === 'FINISHED'" :lang="language" :games="settlementGames" :teams="{yellow:{name:room.schedule?.yellow_name || t('黄方')},white:{name:room.schedule?.white_name || t('白方')}}" :score="match.series_score" :points="match.series_points" :winner="match.winner_side" :reason="match.finish_reason">
          <button class="secondary-button" type="button" @click="navigate(room.event ? `/events/${room.event.slug}` : competitionHomePath)">{{ $t("返回比赛列表") }}</button>
        </MatchSettlement>
        <div v-if="isGameStage && !isGamePlaying" class="phase-language"><LanguageSwitch /></div>
        </div>
        </Transition>
      </div>
    </main>
  </div>
</template>
<style scoped>
.schedule-room-notice{padding:18px;margin-bottom:18px}.schedule-room-notice h2{font-size:clamp(18px,2vw,24px);margin:0 0 12px;line-height:1.5}.schedule-room-notice p{line-height:1.7;margin:10px 0}.schedule-room-notice p:last-child{margin-bottom:0}
.room-create-disclosure{padding:20px;margin:20px 0}.room-create-disclosure>summary{cursor:pointer}.room-create-disclosure>div{margin-top:20px}
.result-boards{display:grid;grid-template-columns:1fr 1fr;gap:28px;max-width:820px;margin:24px auto}.result-boards>div{min-width:0;display:flex;flex-direction:column;align-items:center;gap:12px}.result-boards :deep(.board){max-width:100%}.public-matchups{padding:12px;text-align:center}.blind-candidate-cards{grid-template-columns:1fr 1fr}.operation-form :deep(.player-avatar){width:32px;height:32px;flex:0 0 32px}
@media(max-width:600px){.result-boards{gap:12px}.result-boards>div{font-size:12px}}
</style>
