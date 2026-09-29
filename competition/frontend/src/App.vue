<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { api, connectRoom } from './api';
import { userFacingError } from './errorMessages.js';
import ProjectPlayground from './projects/ProjectPlayground.vue';
import TournamentBoard from './projects/TournamentBoard.vue';
import CargoBoard from './projects/CargoBoard.vue';
import { projectIconUrl } from '../../shared/projectIcons.js';
import {
  competitionProjectInput,
  PROJECT_BY_ID,
  PROJECT_BY_ORDER,
  TOURNAMENT_PROJECTS,
} from './projects/catalog.js';

const pathname = ref(window.location.pathname);
const projectRouteMatch = computed(() => pathname.value.match(/^\/projects(?:\/([^/]+))?\/?$/));
const practiceRouteMatch = computed(() => pathname.value.match(/^\/practice(?:\/(1[0-5]|[1-9]))?\/?$/));
const isProjectRoute = computed(() => Boolean(projectRouteMatch.value || practiceRouteMatch.value));
const projectRouteId = computed(() => (projectRouteMatch.value?.[1]
  ? decodeURIComponent(projectRouteMatch.value[1])
  : PROJECT_BY_ORDER[Number(practiceRouteMatch.value?.[1])]?.id || ''));

const session = ref(null);
const rooms = ref([]);
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
const serverOffsetMs = ref(0);
const movePending = ref(false);
const projectThinking = ref(false);
const mainSiteUrl = String(import.meta.env.VITE_MAIN_SITE_URL || 'https://2048tables.online/');
const competitionHomePath = String(import.meta.env.VITE_COMPETITION_HOME_PATH || '/test');
let disconnectRoom = null;
let ticker = null;
let routeEpoch = 0;
let previousPhaseToken = '';
let previousLineupToken = '';
let previousResultKey = '';
let projectThinkingTimer = null;

const statusText = {
  SEATING: '选手落座',
  READY_CHECK: '队长准备',
  DRAW: '先后手抽签',
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
};

const currentCode = computed(() => {
  const match = pathname.value.match(/^\/rooms\/([A-Za-z0-9]+)\/?$/);
  return match ? match[1].toUpperCase() : '';
});
const selectedProjects = computed(() => selectedProjectIds.value.map(id => PROJECT_BY_ID[id]).filter(Boolean));
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
const allSeated = computed(() => (room.value?.seats?.length || 0) === 6);
const isLobby = computed(() => ['SEATING', 'READY_CHECK'].includes(room.value?.status));
const draft = computed(() => room.value?.draft || null);
const lineup = computed(() => room.value?.lineup || null);
const match = computed(() => room.value?.match || null);
const isGameReady = computed(() => /^GAME_[ABC]_READY$/.test(room.value?.status || ''));
const isGamePlaying = computed(() => /^GAME_[ABC]_PLAYING$/.test(room.value?.status || ''));
const isGameResult = computed(() => /^GAME_[ABC]_RESULT$/.test(room.value?.status || ''));
const remainingSeconds = computed(() => {
  const deadline = Date.parse(
    (room.value?.status === 'LINEUP' ? lineup.value?.deadline_at : draft.value?.deadline_at) || '',
  );
  if (!Number.isFinite(deadline)) return null;
  return Math.max(0, Math.ceil((deadline - (clockNow.value + serverOffsetMs.value)) / 1000));
});
const canSubmitPickBan = computed(() => (
  room.value?.me?.can_submit_pick_ban
  && selectedPick.value
  && selectedBan.value
  && selectedPick.value !== selectedBan.value
));
const canSubmitBlind = computed(() => room.value?.me?.can_submit_blind && selectedBlind.value);
const canSubmitLineup = computed(() => (
  room.value?.me?.can_submit_lineup
  && Object.keys(lineupSelections.value).length === 3
  && new Set(Object.values(lineupSelections.value).map(Number)).size === 3
));

function applyRoom(nextRoom) {
  if (!nextRoom?.room_code || nextRoom.room_code.toUpperCase() !== currentCode.value) return false;
  if (room.value?.room_code === nextRoom.room_code) {
    const currentSequence = Number(room.value.event_sequence);
    const nextSequence = Number(nextRoom.event_sequence);
    const currentVersion = Number(room.value.version);
    const nextVersion = Number(nextRoom.version);
    if (Number.isFinite(currentSequence) && Number.isFinite(nextSequence)
      && (nextSequence < currentSequence
        || (nextSequence === currentSequence && nextVersion < currentVersion))) return false;
  }
  const nextToken = String(nextRoom?.draft?.phase_token || '');
  if (nextToken !== previousPhaseToken) {
    selectedPick.value = '';
    selectedBan.value = '';
    selectedBlind.value = '';
    previousPhaseToken = nextToken;
  }
  const nextLineupToken = String(nextRoom?.lineup?.phase_token || '');
  if (nextLineupToken && nextLineupToken !== previousLineupToken) {
    lineupSelections.value = { A: 1, B: 2, C: 3 };
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
  const serverTime = Date.parse(nextRoom?.server_time || '');
  if (Number.isFinite(serverTime)) serverOffsetMs.value = serverTime - Date.now();
  room.value = nextRoom;
  return true;
}

function setError(cause) {
  error.value = userFacingError(cause);
}

async function loadDashboard(epoch) {
  room.value = null;
  disconnectRoom?.();
  disconnectRoom = null;
  connection.value = 'offline';
  const payload = await api.list();
  if (epoch !== routeEpoch || currentCode.value) return;
  rooms.value = payload.competitions || [];
}

async function loadRoom(code, epoch) {
  disconnectRoom?.();
  disconnectRoom = null;
  room.value = null;
  connection.value = 'connecting';
  const payload = await api.room(code);
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
        applyRoom(message.data);
        connection.value = 'online';
      }
      if (message?.type === 'error') setError(message.error || '房间连接失败');
    },
  });
}

async function route() {
  const epoch = ++routeEpoch;
  loading.value = true;
  error.value = '';
  try {
    if (isProjectRoute.value) {
      disconnectRoom?.();
      disconnectRoom = null;
      room.value = null;
      connection.value = 'offline';
      return;
    }
    if (!session.value) {
      const payload = await api.session();
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
    if (payload?.competition) applyRoom(payload.competition);
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
    if (projects.length < 5) throw new Error('项目池至少需要 5 个项目。');
    if (new Set(projects.map(project => project.id)).size !== projects.length) {
      throw new Error('项目池不能包含重复项目。');
    }
    const payload = await api.create(
      newRoomName.value,
      projects.map(competitionProjectInput),
    );
    navigate(`/rooms/${payload.competition.room_code}`);
    return payload;
  });
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
  if (!room.value?.me?.can_move || busy.value || movePending.value) return;
  const mySide = match.value?.my_session?.side;
  movePending.value = true;
  projectThinking.value = false;
  window.clearTimeout(projectThinkingTimer);
  if (sessionPayload(mySide)?.evil_spawn) {
    projectThinkingTimer = window.setTimeout(() => {
      if (movePending.value) projectThinking.value = true;
    }, 300);
  }
  try {
    await perform(() => api.moveCurrentGame(
      room.value.room_code,
      direction,
      match.value.phase_token,
    ), { ignoreCodes: ['INVALID_MOVE', 'PROJECT_ALREADY_COMPLETE', 'INVALID_GAME_PHASE'] });
  } finally {
    window.clearTimeout(projectThinkingTimer);
    projectThinking.value = false;
    movePending.value = false;
  }
}

function projectAction(action) {
  if (!room.value?.me?.can_move || busy.value) return;
  perform(() => api.actionCurrentGame(
    room.value.room_code,
    action,
    match.value.phase_token,
  ));
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
  return room.value?.projects?.find((project) => project.key === key)?.name || key || '—';
}

function roomProjectIcon(key) {
  return projectIconUrl(room.value?.projects?.find((project) => project.key === key)?.project_ref);
}

function sessionPayload(side) {
  return match.value?.sessions?.[side]?.public_view?.payload || null;
}

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
  const payload = sessionPayload(side) || {};
  let elapsed = Math.max(0, Number(payload.elapsed_ms || 0));
  const sessionState = match.value?.sessions?.[side];
  const snapshotAt = Date.parse(room.value?.server_time || '');
  if (!sessionState?.finished && !match.value?.suspension?.active && Number.isFinite(snapshotAt)) {
    elapsed += Math.max(0, clockNow.value + serverOffsetMs.value - snapshotAt);
  }
  return elapsed;
}

function projectRemainingMs(side) {
  const limit = Number(sessionPayload(side)?.time_limit_ms || 0);
  return limit ? Math.max(0, limit - projectElapsedMs(side)) : null;
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

function sideName(side) {
  return side === 'yellow' ? '黄方' : side === 'white' ? '白方' : '—';
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
  return draft.value?.[`project_${String(game).toLowerCase()}`] || '';
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
  const clock = match.value?.clocks?.[side];
  if (!clock) return 0;
  const deadline = Date.parse(clock.deadline_at || '');
  if (clock.state === 'running' && Number.isFinite(deadline)) {
    return Math.max(0, deadline - (clockNow.value + serverOffsetMs.value));
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
  if (!room.value?.me?.can_move || busy.value || movePending.value) return;
  if (event.defaultPrevented || event.isComposing || event.keyCode === 229) return;
  const target = event.target;
  if (target instanceof Element && target.closest('input, textarea, select, [contenteditable]')) return;
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

function canClaim(seat) {
  return !seat && room.value?.me?.can_claim_seat && !busy.value;
}

function teamSeats(side) {
  return [1, 2, 3].map((position) => ({
    side,
    position,
    seat: seats.value.get(`${side}:${position}`),
  }));
}

onMounted(() => {
  window.addEventListener('popstate', handleLocationChange);
  window.addEventListener('keydown', handleGameKeys);
  ticker = window.setInterval(() => { clockNow.value = Date.now(); }, 16);
  route();
});

onBeforeUnmount(() => {
  ++routeEpoch;
  window.removeEventListener('popstate', handleLocationChange);
  window.removeEventListener('keydown', handleGameKeys);
  window.clearInterval(ticker);
  window.clearTimeout(projectThinkingTimer);
  disconnectRoom?.();
});
</script>

<template>
  <ProjectPlayground v-if="isProjectRoute" :project-id="projectRouteId" />
  <div v-else class="app-shell">
    <header class="site-header">
      <button class="brand" type="button" @click="navigate(competitionHomePath)">
        <span class="brand-mark">20</span>
        <span><strong>2048 赛事中心</strong><small>Competition</small></span>
      </button>
      <div v-if="session?.user" class="account-chip">
        <span class="account-dot"></span>
        {{ session.user.display_name }}
      </div>
    </header>

    <main v-if="loading" class="center-state">
      <span class="spinner"></span>
      <p>正在同步比赛状态…</p>
    </main>

    <main v-else-if="error && !session" class="center-state error-state">
      <h1>需要登录</h1>
      <p>{{ error }}</p>
      <a class="primary-button" :href="mainSiteUrl">前往主站登录</a>
    </main>

    <main v-else-if="!room" class="dashboard page-width">
      <section class="hero-row">
        <div>
          <p class="eyebrow">独立比赛系统</p>
          <h1>比赛房间</h1>
          <p class="muted">创建房间或输入房间码进入候场。观众请从直播大厅进入。</p>
        </div>
        <form class="join-box" @submit.prevent="enterRoom">
          <label for="room-code">房间码</label>
          <div class="inline-form">
            <input id="room-code" v-model="joinCode" maxlength="12" placeholder="例如 K8F3QX" />
            <button class="primary-button" type="submit">进入房间</button>
          </div>
        </form>
      </section>

      <p v-if="error" class="alert">{{ error }}</p>

      <section v-if="session?.can_create_competition" class="panel create-panel">
        <div>
          <p class="eyebrow">举办方</p>
          <h2>创建新比赛</h2>
          <p class="muted">从已注册玩法中选择至少 5 项；创建后规则与顺序固定。</p>
        </div>
        <form class="create-form" @submit.prevent="createRoom">
          <label class="create-name">比赛名称<input v-model="newRoomName" minlength="2" maxlength="100" required placeholder="比赛名称" /></label>
          <fieldset class="project-picker">
            <legend>项目池 · 已选 {{ selectedProjects.length }} 项</legend>
            <p class="muted">勾选项目后，可调整其在 BP 项目池中的顺序。</p>
            <div v-for="project in orderedProjectOptions" :key="project.id" class="project-picker-row">
              <label><input v-model="selectedProjectIds" type="checkbox" :value="project.id" /><img class="project-icon picker-icon" :src="projectIconUrl(project.id)" alt="" /><span><strong>{{ project.title }}</strong><small>{{ project.description }}</small></span></label>
              <div class="project-picker-actions">
                <a :href="project.practicePath" target="_blank" rel="noopener noreferrer">试玩</a>
                <button type="button" :disabled="!selectedProjectIds.includes(project.id) || selectedProjectIds.indexOf(project.id) === 0" :aria-label="`上移 ${project.title}`" @click="moveProjectOrder(project.id, -1)">↑</button>
                <button type="button" :disabled="!selectedProjectIds.includes(project.id) || selectedProjectIds.indexOf(project.id) === selectedProjectIds.length - 1" :aria-label="`下移 ${project.title}`" @click="moveProjectOrder(project.id, 1)">↓</button>
                <span v-if="selectedProjectIds.includes(project.id)" class="project-picker-order">{{ selectedProjectIds.indexOf(project.id) + 1 }}</span>
              </div>
            </div>
          </fieldset>
          <p v-if="selectedProjects.length < 5" class="alert">项目池至少需要 5 项。</p>
          <button class="primary-button" type="submit" :disabled="busy || selectedProjects.length < 5">创建房间</button>
        </form>
      </section>

      <section class="room-list-section">
        <div class="section-heading"><h2>我的比赛</h2><span>{{ rooms.length }} 场</span></div>
        <div v-if="rooms.length" class="room-list">
          <button v-for="item in rooms" :key="item.id" class="room-row" type="button" @click="navigate(`/rooms/${item.room_code}`)">
            <span><strong>{{ item.name }}</strong><small>{{ item.room_code }}</small></span>
            <span class="room-meta">{{ item.occupied_seats }}/6 · {{ statusText[item.status] || item.status }}</span>
          </button>
        </div>
        <p v-else class="empty-state">暂无与你相关的比赛房间。</p>
      </section>
    </main>

    <main v-else class="room-page">
      <section class="match-header">
        <div class="page-width match-header-inner">
          <button class="back-button" type="button" @click="navigate(competitionHomePath)">← 比赛列表</button>
          <div class="match-identity">
            <span class="room-code">{{ room.room_code }}</span>
            <h1>{{ room.name }}</h1>
          </div>
          <div class="stage-chip"><span :class="['connection-dot', connection]"></span>{{ statusText[room.status] || room.status }}</div>
        </div>
      </section>

      <section class="progress-strip">
        <div class="progress-step done">房间创建</div>
        <div :class="['progress-step', room.status === 'SEATING' ? 'active' : 'done']">选手落座</div>
        <div :class="['progress-step', room.status === 'READY_CHECK' ? 'active' : !['SEATING', 'READY_CHECK'].includes(room.status) ? 'done' : '']">队长准备</div>
        <div :class="['progress-step', room.status === 'DRAW' ? 'active' : !['SEATING', 'READY_CHECK', 'DRAW'].includes(room.status) ? 'done' : '']">先后手抽签</div>
        <div :class="['progress-step', ['FIRST_PICK_BAN', 'SECOND_PICK_BAN', 'BLIND_PICK'].includes(room.status) ? 'active' : ['C_DRAW', 'LINEUP'].includes(room.status) || room.status.startsWith('GAME_') || room.status === 'FINISHED' ? 'done' : '']">项目 BP</div>
        <div :class="['progress-step', room.status === 'LINEUP' ? 'active' : room.status.startsWith('GAME_') || room.status === 'FINISHED' ? 'done' : '']">秘密布阵</div>
        <div :class="['progress-step', room.status.startsWith('GAME_') ? 'active' : room.status === 'FINISHED' ? 'done' : '']">项目对局</div>
      </section>

      <div class="page-width room-content">
        <p v-if="error" class="alert">{{ error }}</p>

        <section v-if="match?.suspension?.active" class="suspension-banner">
          <div class="suspension-copy">
            <p class="eyebrow">MATCH SUSPENDED</p>
            <h2>比赛已由赛事方暂停</h2>
            <p>{{ match.suspension.reason_text }}</p>
            <small>{{ match.suspension.started_by_display_name || `赛事方 #${match.suspension.started_by_user_id}` }} · 两队计时器已冻结</small>
          </div>
          <div class="resume-status">
            <span :class="match.suspension.resume_readiness.yellow.ready && 'ready'">黄队长 {{ match.suspension.resume_readiness.yellow.ready ? '已就绪' : '待就绪' }}</span>
            <span :class="match.suspension.resume_readiness.white.ready && 'ready'">白队长 {{ match.suspension.resume_readiness.white.ready ? '已就绪' : '待就绪' }}</span>
          </div>
          <div class="suspension-actions">
            <button v-if="room.me.can_mark_resume_ready" class="secondary-button" type="button" :disabled="busy" @click="toggleResumeReadiness">
              {{ match.suspension.resume_readiness[mySeat.side].ready ? '取消恢复就绪' : '队长确认可恢复' }}
            </button>
            <button v-if="room.me.is_match_official" class="primary-button" type="button" :disabled="busy || !room.me.can_resume" @click="resumeMatch">恢复比赛与计时</button>
          </div>
        </section>

        <section v-if="match && room.status.startsWith('GAME_')" class="match-operations">
          <details v-if="room.me.can_report_issue || room.issues.length" class="operation-card">
            <summary>问题上报 <span v-if="room.issues.length">{{ room.issues.length }} 项待处理</span></summary>
            <div v-if="room.me.can_report_issue" class="operation-form issue-form">
              <select v-model="issueCategory" aria-label="问题类别">
                <option value="network_device">网络 / 设备</option>
                <option value="project">项目运行</option>
                <option value="rules">规则争议</option>
                <option value="other">其他</option>
              </select>
              <input v-model="issueDetails" maxlength="500" placeholder="简要描述问题（至少 3 个字符）" />
              <button class="secondary-button" type="button" :disabled="busy || issueDetails.trim().length < 3" @click="reportIssue">提交给裁判</button>
            </div>
            <div v-if="room.issues.length" class="issue-list">
              <article v-for="issue in room.issues" :key="issue.id">
                <div><strong>{{ issue.reporter_display_name }}</strong><span>{{ issue.details }}</span><small>{{ issue.category }}</small></div>
                <div v-if="room.me.is_match_official" class="issue-resolution">
                  <input v-model="issueResolutionNote" maxlength="500" placeholder="处理说明" />
                  <button type="button" :disabled="busy || issueResolutionNote.trim().length < 3" @click="resolveIssue(issue)">解决</button>
                  <button type="button" :disabled="busy || issueResolutionNote.trim().length < 3" @click="resolveIssue(issue, 'dismissed')">驳回</button>
                </div>
              </article>
            </div>
          </details>

          <details v-if="room.me.is_match_official" class="operation-card referee-card">
            <summary>裁判控制台 <span>所有操作写入审计事件</span></summary>
            <div class="referee-grid">
              <form v-if="room.me.can_suspend" class="operation-form" @submit.prevent="suspendMatch">
                <strong>暂停比赛</strong>
                <select v-model="suspendReasonCode">
                  <option value="network_device">网络 / 设备</option>
                  <option value="project">项目故障</option>
                  <option value="rules">规则争议</option>
                  <option value="medical">医疗情况</option>
                  <option value="other">其他</option>
                </select>
                <input v-model="suspendReasonText" maxlength="500" placeholder="公开暂停原因" />
                <button class="danger-button" type="submit" :disabled="busy || suspendReasonText.trim().length < 3">暂停并冻结计时</button>
              </form>

              <form v-if="room.me.can_override_result" class="operation-form result-correction" @submit.prevent="overrideResult">
                <strong>纠正当前赛果</strong>
                <label>黄方分数 <input v-model.number="overrideYellowScore" type="number" min="0" /></label>
                <label>白方分数 <input v-model.number="overrideWhiteScore" type="number" min="0" /></label>
                <select v-model="overrideWinner">
                  <option value="yellow">黄方胜</option><option value="white">白方胜</option><option value="draw">平局</option>
                </select>
                <input v-model="overrideReason" maxlength="500" placeholder="纠正原因" />
                <button class="secondary-button" type="submit" :disabled="busy || overrideReason.trim().length < 3">发布修订并清空确认</button>
              </form>

              <form v-if="room.me.can_force_advance" class="operation-form" @submit.prevent="forceAdvance">
                <strong>强制推进</strong>
                <input v-model="forceAdvanceReason" maxlength="500" placeholder="跳过队长确认的原因" />
                <button class="secondary-button" type="submit" :disabled="busy || forceAdvanceReason.trim().length < 3">进入下一项目</button>
              </form>

              <form v-if="room.me.can_force_finish" class="operation-form" @submit.prevent="forceFinish">
                <strong>强制结束全场</strong>
                <select v-model="forceFinishWinner"><option value="yellow">黄方胜</option><option value="white">白方胜</option><option value="draw">平局</option></select>
                <input v-model="forceFinishReason" maxlength="500" placeholder="裁决原因" />
                <button class="danger-button" type="submit" :disabled="busy || forceFinishReason.trim().length < 3">结束并冻结结果</button>
              </form>
            </div>
          </details>
        </section>

        <template v-if="isLobby">
          <section class="teams-grid">
            <div class="team-panel yellow-team">
              <div class="team-heading">
                <div><p class="eyebrow">YELLOW SIDE</p><h2>黄队</h2></div>
                <span :class="['ready-chip', room.teams.yellow.ready && 'is-ready']">{{ room.teams.yellow.ready ? '已准备' : '未准备' }}</span>
              </div>
              <button
                v-for="item in teamSeats('yellow')"
                :key="item.position"
                :class="['seat-card', item.seat && 'occupied', item.seat?.user_id === room.me.user_id && 'mine']"
                type="button"
                :disabled="!canClaim(item.seat)"
                @click="claim(item.side, item.position)"
              >
                <span class="seat-number">{{ seatLabel(item.side, item.position) }}</span>
                <span v-if="item.seat" class="seat-player"><strong>{{ item.seat.display_name }}</strong><small>{{ item.position === 1 ? '队长席位' : '队员席位' }}</small></span>
                <span v-else class="seat-empty"><strong>空位</strong><small>{{ room.me.can_claim_seat ? '点击落座' : '等待选手' }}</small></span>
                <span v-if="item.seat?.user_id === room.me.user_id" class="you-label">你</span>
              </button>
            </div>

            <div class="versus-column">
              <span class="versus">VS</span>
              <span>{{ room.seats.length }} / 6 已落座</span>
              <span v-if="room.status === 'SEATING'">等待全部选手</span>
              <span v-else>等待双方队长</span>
            </div>

            <div class="team-panel white-team">
              <div class="team-heading">
                <div><p class="eyebrow">WHITE SIDE</p><h2>白队</h2></div>
                <span :class="['ready-chip', room.teams.white.ready && 'is-ready']">{{ room.teams.white.ready ? '已准备' : '未准备' }}</span>
              </div>
              <button
                v-for="item in teamSeats('white')"
                :key="item.position"
                :class="['seat-card', item.seat && 'occupied', item.seat?.user_id === room.me.user_id && 'mine']"
                type="button"
                :disabled="!canClaim(item.seat)"
                @click="claim(item.side, item.position)"
              >
                <span class="seat-number">{{ seatLabel(item.side, item.position) }}</span>
                <span v-if="item.seat" class="seat-player"><strong>{{ item.seat.display_name }}</strong><small>{{ item.position === 1 ? '队长席位' : '队员席位' }}</small></span>
                <span v-else class="seat-empty"><strong>空位</strong><small>{{ room.me.can_claim_seat ? '点击落座' : '等待选手' }}</small></span>
                <span v-if="item.seat?.user_id === room.me.user_id" class="you-label">你</span>
              </button>
            </div>
          </section>

          <section class="action-bar">
            <div>
              <strong v-if="mySeat">你位于 {{ seatLabel(mySeat.side, mySeat.position) }}{{ mySeat.position === 1 ? '，是本队队长' : '' }}</strong>
              <strong v-else>请选择一个空位落座</strong>
              <p v-if="room.status === 'SEATING'">六个席位坐满后，双方队长可以准备。</p>
              <p v-else>任一队准备后席位将锁定，双方准备后进入抽签。</p>
            </div>
            <div class="action-buttons">
              <button v-if="room.me.can_leave_seat" class="secondary-button" type="button" :disabled="busy" @click="leave">离开席位</button>
              <button v-if="room.me.can_ready" class="primary-button" type="button" :disabled="busy || !allSeated" @click="toggleReady">{{ myTeamReady ? '取消准备' : '队长准备' }}</button>
            </div>
          </section>
        </template>

        <section v-else-if="room.status === 'DRAW'" class="draw-stage">
          <p class="eyebrow">FIRST SIDE DRAW</p>
          <div class="draw-orbit"><span>{{ sideName(draft.first_side).slice(0, 1) }}</span></div>
          <h1>{{ sideName(draft.first_side) }}获得先手</h1>
          <p>{{ captainName(draft.first_side) }} 将首先选择项目 A 并 BAN 一个项目</p>
          <strong class="draw-countdown">{{ formatCountdown(remainingSeconds) }}</strong>
          <small>抽签承诺 {{ draft.commitment.slice(0, 12) }}… · {{ draft.algorithm_version }}</small>
        </section>

        <template v-else-if="['FIRST_PICK_BAN', 'SECOND_PICK_BAN', 'BLIND_PICK'].includes(room.status)">
          <section class="bp-team-bar">
            <div :class="['bp-team-side', 'yellow', draft.active_side === 'yellow' && 'is-active']">
              <span>黄方</span><strong>{{ captainName('yellow') }}</strong>
              <small>{{ draft.first_side === 'yellow' ? '先手' : '后手' }}</small>
            </div>
            <div class="bp-phase-title">
              <span>{{ statusText[room.status] }}</span>
              <strong>{{ formatCountdown(remainingSeconds) }}</strong>
              <small v-if="room.status !== 'BLIND_PICK'">{{ sideName(draft.active_side) }}队长操作</small>
              <small v-else>双方独立提交，选择互不可见</small>
            </div>
            <div :class="['bp-team-side', 'white', draft.active_side === 'white' && 'is-active']">
              <span>白方</span><strong>{{ captainName('white') }}</strong>
              <small>{{ draft.first_side === 'white' ? '先手' : '后手' }}</small>
            </div>
          </section>

          <section class="draft-history">
            <div><img v-if="roomProjectIcon(draft.project_a)" class="project-icon history-icon" :src="roomProjectIcon(draft.project_a)" alt="" /><span>项目 A</span><strong>{{ projectName(draft.project_a) }}</strong></div>
            <div class="ban"><img v-if="roomProjectIcon(draft.ban_m)" class="project-icon history-icon" :src="roomProjectIcon(draft.ban_m)" alt="" /><span>BAN M</span><strong>{{ projectName(draft.ban_m) }}</strong></div>
            <div><img v-if="roomProjectIcon(draft.project_b)" class="project-icon history-icon" :src="roomProjectIcon(draft.project_b)" alt="" /><span>项目 B</span><strong>{{ projectName(draft.project_b) }}</strong></div>
            <div class="ban"><img v-if="roomProjectIcon(draft.ban_n)" class="project-icon history-icon" :src="roomProjectIcon(draft.ban_n)" alt="" /><span>BAN N</span><strong>{{ projectName(draft.ban_n) }}</strong></div>
          </section>

          <section v-if="room.status !== 'BLIND_PICK'" class="bp-workspace">
            <div class="bp-instruction">
              <div>
                <p class="eyebrow">PICK + BAN</p>
                <h2 v-if="room.me.can_submit_pick_ban">选择一个比赛项目，并 BAN 一个不同项目</h2>
                <h2 v-else>等待 {{ sideName(draft.active_side) }}队长完成选择</h2>
              </div>
              <div v-if="room.me.can_submit_pick_ban" class="current-choices">
                <span>PICK <strong>{{ projectName(selectedPick) }}</strong></span>
                <span>BAN <strong>{{ projectName(selectedBan) }}</strong></span>
              </div>
            </div>
            <div class="project-pool">
              <article
                v-for="project in room.projects"
                :key="project.key"
                :class="['project-card', !isProjectAvailable(project.key) && 'unavailable', selectedPick === project.key && 'selected-pick', selectedBan === project.key && 'selected-ban']"
              >
                <span class="project-order">{{ String(project.sort_order).padStart(2, '0') }}</span>
                <img v-if="projectIconUrl(project.project_ref)" class="project-icon pool-icon" :src="projectIconUrl(project.project_ref)" alt="" />
                <h3>{{ project.name }}</h3>
                <p>{{ project.description || `规则版本 ${project.rules_version}` }}</p>
                <div v-if="room.me.can_submit_pick_ban && isProjectAvailable(project.key)" class="project-actions">
                  <button type="button" :class="selectedPick === project.key && 'active'" @click="choosePick(project.key)">选择</button>
                  <button type="button" :class="['ban-action', selectedBan === project.key && 'active']" @click="chooseBan(project.key)">BAN</button>
                </div>
                <span v-else-if="draft.project_a === project.key" class="project-result">项目 A</span>
                <span v-else-if="draft.project_b === project.key" class="project-result">项目 B</span>
                <span v-else-if="[draft.ban_m, draft.ban_n].includes(project.key)" class="project-result banned">已 BAN</span>
              </article>
            </div>
            <div v-if="room.me.can_submit_pick_ban" class="draft-submit-bar">
              <span>超时后系统按项目池顺序选择首个合法组合。</span>
              <button class="primary-button" type="button" :disabled="busy || !canSubmitPickBan" @click="submitPickBan">锁定并公开</button>
            </div>
          </section>

          <section v-else class="bp-workspace blind-workspace">
            <div class="bp-instruction">
              <div><p class="eyebrow">SEALED PICK</p><h2>双方同时盲选项目 C 候选</h2></div>
              <div class="blind-status">
                <span :class="draft.blind_submissions.yellow && 'submitted'">黄方 {{ draft.blind_submissions.yellow ? '已提交' : '选择中' }}</span>
                <span :class="draft.blind_submissions.white && 'submitted'">白方 {{ draft.blind_submissions.white ? '已提交' : '选择中' }}</span>
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
                <span class="project-order">{{ String(project.sort_order).padStart(2, '0') }}</span>
                <img v-if="projectIconUrl(project.project_ref)" class="project-icon pool-icon" :src="projectIconUrl(project.project_ref)" alt="" />
                <h3>{{ project.name }}</h3>
                <p>{{ project.description || `规则版本 ${project.rules_version}` }}</p>
              </button>
            </div>
            <div class="draft-submit-bar">
              <span v-if="draft.my_blind_choice">本队已密封提交：{{ projectName(draft.my_blind_choice) }}</span>
              <span v-else-if="!room.me.can_submit_blind">等待双方队长提交；候选项目不会提前公开。</span>
              <span v-else>选定后不可修改；超时使用第一个合法项目。</span>
              <button v-if="room.me.can_submit_blind" class="primary-button" type="button" :disabled="busy || !canSubmitBlind" @click="submitBlind">密封提交</button>
            </div>
          </section>
        </template>

        <section v-else-if="room.status === 'C_DRAW'" class="draft-complete">
          <p class="eyebrow">DRAFT COMPLETE</p>
          <h1>三个比赛项目已确定</h1>
          <div class="final-projects">
            <div><img v-if="roomProjectIcon(draft.project_a)" class="project-icon summary-icon" :src="roomProjectIcon(draft.project_a)" alt="" /><span>A · {{ sideName(draft.first_side) }}选择</span><strong>{{ projectName(draft.project_a) }}</strong></div>
            <div><img v-if="roomProjectIcon(draft.project_b)" class="project-icon summary-icon" :src="roomProjectIcon(draft.project_b)" alt="" /><span>B · {{ sideName(draft.second_side) }}选择</span><strong>{{ projectName(draft.project_b) }}</strong></div>
            <div class="project-c"><img v-if="roomProjectIcon(draft.project_c)" class="project-icon summary-icon" :src="roomProjectIcon(draft.project_c)" alt="" /><span>C · 盲选抽签</span><strong>{{ projectName(draft.project_c) }}</strong></div>
          </div>
          <div class="blind-reveal">
            <span>黄方候选：<strong>{{ projectName(draft.blind_choices.yellow) }}</strong></span>
            <span>白方候选：<strong>{{ projectName(draft.blind_choices.white) }}</strong></span>
            <span>BAN：{{ projectName(draft.ban_m) }} / {{ projectName(draft.ban_n) }}</span>
          </div>
          <p>BP 流程已完成，{{ formatCountdown(remainingSeconds) }} 后进入双方队长秘密布阵。</p>
        </section>

        <template v-else-if="room.status === 'LINEUP'">
          <section class="bp-team-bar lineup-team-bar">
            <div class="bp-team-side yellow">
              <span>黄方</span><strong>{{ captainName('yellow') }}</strong>
              <small :class="['submission-state', lineup.submissions.yellow && 'submitted']">{{ lineup.submissions.yellow ? '阵容已密封' : '队长布阵中' }}</small>
            </div>
            <div class="bp-phase-title">
              <span>SECRET LINEUP</span>
              <strong>{{ formatCountdown(remainingSeconds) }}</strong>
              <small>双方独立计时 · 完整编排仅本队队员可见</small>
            </div>
            <div class="bp-team-side white">
              <span>白方</span><strong>{{ captainName('white') }}</strong>
              <small :class="['submission-state', lineup.submissions.white && 'submitted']">{{ lineup.submissions.white ? '阵容已密封' : '队长布阵中' }}</small>
            </div>
          </section>

          <section class="lineup-projects">
            <div v-for="game in ['A', 'B', 'C']" :key="game">
              <img v-if="roomProjectIcon(gameProject(game))" class="project-icon lineup-icon" :src="roomProjectIcon(gameProject(game))" alt="" /><span>第 {{ game }} 场</span><strong>{{ projectName(gameProject(game)) }}</strong>
            </div>
          </section>

          <section class="lineup-workspace">
            <div class="lineup-roster yellow-roster">
              <p class="eyebrow">YELLOW TEAM</p>
              <h2>黄方名单</h2>
              <div v-for="item in teamSeats('yellow')" :key="item.position" class="lineup-player">
                <span>{{ item.position }}</span><strong>{{ item.seat?.display_name }}</strong><small>{{ item.position === 1 ? '队长' : '队员' }}</small>
              </div>
            </div>

            <div class="lineup-console">
              <template v-if="room.me.can_submit_lineup">
                <div class="lineup-title"><p class="eyebrow">CAPTAIN ONLY</p><h2>安排三场出战顺序</h2><p>每名队员必须且只能出战一场。提交后不可修改。</p></div>
                <label v-for="game in ['A', 'B', 'C']" :key="game" class="lineup-assignment">
                  <span><b>{{ game }}</b><img v-if="roomProjectIcon(gameProject(game))" class="project-icon assignment-icon" :src="roomProjectIcon(gameProject(game))" alt="" /><small>{{ projectName(gameProject(game)) }}</small></span>
                  <select v-model.number="lineupSelections[game]">
                    <option v-for="position in [1, 2, 3]" :key="position" :value="position">{{ position }} 号位 · {{ playerAt(mySeat.side, position) }}</option>
                  </select>
                </label>
                <p v-if="!canSubmitLineup" class="lineup-warning">同一名队员不能重复出战，请为三场选择不同席位。</p>
                <button class="primary-button lineup-submit" type="button" :disabled="busy || !canSubmitLineup" @click="submitLineup">密封提交阵容</button>
                <small class="lineup-timeout">超时将自动采用 A=1、B=2、C=3。</small>
              </template>
              <template v-else-if="lineup.my_lineup">
                <div class="sealed-mark">✓</div>
                <h2>本队阵容已密封</h2>
                <p>完整编排仅本队三名队员可见。你可以核对本队安排。</p>
                <div v-for="game in ['A', 'B', 'C']" :key="game" class="sealed-row">
                  <b>{{ game }}</b><span>{{ projectName(gameProject(game)) }}</span><strong>{{ lineup.my_lineup[game].display_name }}</strong>
                </div>
              </template>
              <template v-else>
                <div class="sealed-mark waiting">•••</div>
                <h2>{{ room.me.is_captain ? '等待对方队长提交' : '队长正在秘密布阵' }}</h2>
                <p>提交状态全场可见；完整编排仅各队队员可见，当场出战者在该场开赛时公布。</p>
              </template>
            </div>

            <div class="lineup-roster white-roster">
              <p class="eyebrow">WHITE TEAM</p>
              <h2>白方名单</h2>
              <div v-for="item in teamSeats('white')" :key="item.position" class="lineup-player">
                <span>{{ item.position }}</span><strong>{{ item.seat?.display_name }}</strong><small>{{ item.position === 1 ? '队长' : '队员' }}</small>
              </div>
            </div>
          </section>
        </template>

        <template v-else-if="isGameReady">
          <section class="game-scorebar">
            <div class="team-clock yellow-clock"><span>黄方包干时间</span><strong>{{ formatTeamClock(teamClockMs('yellow')) }}</strong></div>
            <div class="current-game-title"><span>GAME {{ match.current_game_key }}</span><img v-if="roomProjectIcon(match.project_key)" class="project-icon current-icon" :src="roomProjectIcon(match.project_key)" alt="" /><strong>{{ projectName(match.project_key) }}</strong><small>系列比分 {{ match.series_score.yellow }} : {{ match.series_score.white }}</small></div>
            <div class="team-clock white-clock"><span>白方包干时间</span><strong>{{ formatTeamClock(teamClockMs('white')) }}</strong></div>
          </section>
          <section class="pregame-panel">
            <div class="pregame-heading"><p class="eyebrow">PRE-GAME CHECK</p><h1>项目 {{ match.current_game_key }} 开局检查</h1><p>双方出战者和队长全部就绪后，项目与两队包干计时自动开始。</p></div>
            <div class="pregame-versus">
              <div v-for="side in ['yellow', 'white']" :key="side" :class="['pregame-team', side]">
                <span class="side-kicker">{{ sideName(side) }}</span>
                <h2>{{ match.players[side]?.display_name || '待开赛公布' }}</h2>
                <small v-if="match.players[side]">{{ match.players[side].position }} 号位 · 本场出战</small>
                <small v-else>本队队员可查看本队安排</small>
                <div class="check-row"><span>出战者连接与就绪</span><b :class="match.readiness[side].player_ready && 'ready'">{{ match.readiness[side].player_ready ? '已就绪' : '未就绪' }}</b></div>
                <div class="check-row"><span>队长确认</span><b :class="match.readiness[side].captain_ready && 'ready'">{{ match.readiness[side].captain_ready ? '已确认' : '未确认' }}</b></div>
              </div>
            </div>
            <div class="pregame-actions">
              <button v-if="room.me.can_mark_player_ready" class="secondary-button" type="button" :disabled="busy" @click="toggleGameReadiness('player')">{{ match.readiness[mySeat.side].player_ready ? '取消出战者就绪' : '我是出战者，已就绪' }}</button>
              <button v-if="room.me.can_mark_captain_ready" class="secondary-button" type="button" :disabled="busy" @click="toggleGameReadiness('captain')">{{ match.readiness[mySeat.side].captain_ready ? '取消队长确认' : '队长确认开局' }}</button>
              <span v-if="!room.me.can_mark_player_ready && !room.me.can_mark_captain_ready">等待双方出战者与队长完成开局检查。</span>
            </div>
          </section>
        </template>

        <template v-else-if="isGamePlaying">
          <section class="game-scorebar live-scorebar">
            <div class="team-clock yellow-clock"><span>黄方 · {{ match.sessions.yellow?.finished ? '已完成' : '进行中' }}</span><strong>{{ formatTeamClock(teamClockMs('yellow')) }}</strong></div>
            <div class="current-game-title"><span>GAME {{ match.current_game_key }} · LIVE</span><img v-if="roomProjectIcon(match.project_key)" class="project-icon current-icon" :src="roomProjectIcon(match.project_key)" alt="" /><strong>{{ projectName(match.project_key) }}</strong><small>系列比分 {{ match.series_score.yellow }} : {{ match.series_score.white }}</small></div>
            <div class="team-clock white-clock"><span>白方 · {{ match.sessions.white?.finished ? '已完成' : '进行中' }}</span><strong>{{ formatTeamClock(teamClockMs('white')) }}</strong></div>
          </section>
          <section class="game-play-layout">
            <div class="game-participants">
              <div v-for="side in ['yellow', 'white']" :key="side" :class="['participant-card', side]">
                <span>{{ sideName(side) }}</span><strong>{{ match.players[side].display_name }}</strong><small>{{ match.sessions[side]?.finished ? '服务器已接受完成' : '项目进行中' }}</small>
              </div>
            </div>
            <div class="project-dual-view">
              <article v-for="side in ['yellow', 'white']" :key="side" :class="['project-side-view', side, match.sessions[side]?.finished && 'finished']">
                <header>
                  <div><p class="eyebrow">{{ side === 'yellow' ? 'YELLOW PROJECT VIEW' : 'WHITE PROJECT VIEW' }}</p><h2>{{ match.players[side].display_name }}</h2></div>
                  <div class="project-score"><span>{{ sessionPayload(side)?.time_limit_ms ? '已送出' : '得分' }}</span><strong>{{ sessionPayload(side)?.score ?? 0 }}</strong></div>
                </header>
                <CargoBoard
                  v-if="match.sessions[side]?.public_view?.view_protocol === 'cargo-transport-v1'"
                  class="embedded-project-board"
                  :snapshot="projectBoardSnapshot(side)"
                  :disabled="!isMyActiveSide(side) || !room.me.can_move || busy || movePending"
                  :aria-label="`${sideName(side)}真华容道棋盘`"
                  @move="moveGame"
                />
                <TournamentBoard
                  v-else-if="['2048-board-v1', '2048-board-v2'].includes(match.sessions[side]?.public_view?.view_protocol)"
                  class="embedded-project-board"
                  :snapshot="projectBoardSnapshot(side)"
                  :mirror-portals="Boolean(sessionPayload(side)?.mirror_portals)"
                  :irregular-shape="Boolean(sessionPayload(side)?.shape_shifter)"
                  :show-dice-effect="Boolean(sessionPayload(side)?.dice)"
                  :disabled="!isMyActiveSide(side) || !room.me.can_move || busy || movePending"
                  :aria-label="`${sideName(side)}项目棋盘`"
                  @move="moveGame"
                />
                <div v-else class="project-view-fallback"><strong>项目公开画面暂不可用</strong><span>{{ match.sessions[side]?.public_view?.view_kind || '等待项目状态' }}</span></div>
                <div v-if="isMyActiveSide(side) && projectThinking" class="project-thinking-pill" role="status"><span></span>AI 思考中</div>
                <footer>
                  <span>{{ match.sessions[side]?.finished ? '服务器已接受完成' : '项目进行中' }}</span>
                  <small>{{ sessionPayload(side)?.move_count ?? 0 }} 步 · {{ projectRemainingMs(side) == null ? formatProjectElapsed(projectElapsedMs(side)) : `剩余 ${formatProjectElapsed(projectRemainingMs(side))}` }}</small>
                </footer>
                <template v-if="isMyActiveSide(side)">
                  <div v-if="room.me.can_move" class="move-pad side-move-pad">
                    <button type="button" aria-label="向上" :disabled="busy || movePending" @click="moveGame('up')">↑</button>
                    <button type="button" aria-label="向左" :disabled="busy || movePending" @click="moveGame('left')">←</button>
                    <button type="button" aria-label="向下" :disabled="busy || movePending" @click="moveGame('down')">↓</button>
                    <button type="button" aria-label="向右" :disabled="busy || movePending" @click="moveGame('right')">→</button>
                  </div>
                  <div v-if="room.me.can_move && (sessionPayload(side)?.allow_undo || sessionPayload(side)?.allow_restart)" class="project-action-row">
                    <button v-if="sessionPayload(side)?.allow_undo" type="button" :disabled="busy || !sessionPayload(side)?.can_undo" @click="projectAction('undo')">撤销一步</button>
                    <button v-if="sessionPayload(side)?.allow_restart" type="button" :disabled="busy" @click="projectAction('restart')">重新开始</button>
                  </div>
                  <p v-if="room.me.can_move" class="move-help">这是你的操作棋盘。方向键或 WASD 操作；对手棋盘仅供查看。</p>
                  <div v-else-if="match.suspension.active" class="project-finished-note suspended-note"><strong>比赛暂停</strong><span>操作已锁定，等待双方队长与裁判恢复比赛。</span></div>
                </template>
              </article>
            </div>
          </section>
        </template>

        <section v-else-if="isGameResult" class="game-result-panel">
          <div class="game-scorebar">
            <div class="team-clock yellow-clock"><span>黄方剩余</span><strong>{{ formatTeamClock(teamClockMs('yellow')) }}</strong></div>
            <div class="current-game-title"><span>GAME {{ match.current_game_key }} · RESULT</span><strong>{{ winnerName(match.current_result.winner_side) }}</strong><small>等待双方队长确认</small></div>
            <div class="team-clock white-clock"><span>白方剩余</span><strong>{{ formatTeamClock(teamClockMs('white')) }}</strong></div>
          </div>
          <div class="result-versus">
            <div class="result-side yellow"><span>黄方 · {{ match.players.yellow.display_name }}</span><strong>{{ match.current_result.yellow_score }}</strong></div>
            <div class="result-center"><img v-if="roomProjectIcon(match.project_key)" class="project-icon result-icon" :src="roomProjectIcon(match.project_key)" alt="" /><span>{{ projectName(match.project_key) }}</span><b>{{ match.current_result.winner_side === 'draw' ? '平局' : `${sideName(match.current_result.winner_side)}获胜` }}</b></div>
            <div class="result-side white"><span>白方 · {{ match.players.white.display_name }}</span><strong>{{ match.current_result.white_score }}</strong></div>
          </div>
          <p v-if="match.current_result.corrected" class="result-correction-note">裁判已修订 · {{ match.current_result.correction_reason }}</p>
          <div class="confirmation-strip">
            <span :class="match.confirmations.yellow && 'confirmed'">黄队长 {{ match.confirmations.yellow ? '已确认' : '待确认' }}</span>
            <button v-if="room.me.can_confirm_result" class="primary-button" type="button" :disabled="busy" @click="confirmResult">确认本局结果</button>
            <strong v-else>双方确认后自动进入下一项目</strong>
            <span :class="match.confirmations.white && 'confirmed'">白队长 {{ match.confirmations.white ? '已确认' : '待确认' }}</span>
          </div>
        </section>

        <section v-else-if="room.status === 'FINISHED'" class="match-finished-panel">
          <p class="eyebrow">MATCH FINISHED</p>
          <h1>{{ match.winner_side === 'draw' ? '全场平局' : `${sideName(match.winner_side)}赢得比赛` }}</h1>
          <div class="final-series-score"><strong>{{ match.series_score.yellow }}</strong><span>黄方&nbsp;&nbsp;—&nbsp;&nbsp;白方</span><strong>{{ match.series_score.white }}</strong></div>
          <div class="finished-results">
            <div v-for="result in match.results" :key="result.game_key">
              <span><img v-if="roomProjectIcon(result.project_key)" class="project-icon finished-icon" :src="roomProjectIcon(result.project_key)" alt="" />项目 {{ result.game_key }} · {{ projectName(result.project_key) }}</span>
              <strong>{{ result.yellow_score }} : {{ result.white_score }}</strong>
              <small>{{ winnerName(result.winner_side) }}{{ result.reason.includes('clock_expired') ? ' · 包干时间耗尽' : '' }}</small>
            </div>
          </div>
          <p>比赛结果已冻结。正式项目规则接入时将继续复用同一状态机与时钟。</p>
        </section>
      </div>
    </main>
  </div>
</template>
