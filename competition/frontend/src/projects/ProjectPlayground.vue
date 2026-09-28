<template>
  <div :class="['project-lab', { 'is-dark': practiceTheme === 'dark' }]">
    <header class="lab-header">
      <a class="lab-brand" href="/practice"><span>20</span><strong>2048 赛事项目试玩</strong></a>
      <nav>
        <a :href="competitionHomePath" aria-label="赛事中心"><span class="full-label">赛事中心</span><span class="compact-label" aria-hidden="true">赛事</span></a>
        <b>tournament.2048tables.online</b>
        <button class="theme-toggle" type="button" :aria-label="practiceTheme === 'dark' ? '切换为浅色模式' : '切换为深色模式'" :aria-pressed="practiceTheme === 'dark'" @click="toggleTheme">
          <svg v-if="practiceTheme === 'dark'" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42"/></svg>
          <svg v-else viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5 8.5 8.5 0 1 0 20.5 14.5Z"/></svg>
          <span>{{ practiceTheme === 'dark' ? '浅色' : '深色' }}</span>
        </button>
      </nav>
    </header>

    <main v-if="!project" class="project-index">
      <section class="index-intro"><p>TOURNAMENT PROJECT LAB</p><h1>比赛项目试玩</h1><span>以下页面用于举办方验收规则、选手熟悉操作。试玩成绩不会进入正式比赛。</span></section>
      <div class="project-list">
        <a v-for="item in projects" :key="item.id" :href="item.practicePath" class="project-entry">
          <img class="project-art entry-art" :src="projectIconUrl(item.id, practiceTheme)" alt="" /><small>PROJECT {{ item.order }}</small><h2>{{ item.title }}</h2><p>{{ item.description }}</p><footer><span>{{ item.boardLabel || `${item.rows}×${item.cols}` }}</span><b>开始试玩 →</b></footer>
        </a>
      </div>
    </main>

    <main v-else class="play-page">
      <aside class="project-rail">
        <a href="/practice" class="rail-back">← 全部项目</a>
        <a v-for="item in projects" :key="item.id" :href="item.practicePath" :class="{ active: item.id === project.id }">
          <img class="project-art rail-art" :src="projectIconUrl(item.id)" alt="" /><small>{{ item.order }}</small><span>{{ item.shortTitle }}</span>
        </a>
      </aside>

      <section class="play-main">
        <header class="project-heading">
          <img class="project-art heading-art" :src="projectIconUrl(project.id, practiceTheme)" alt="" /><div><p>PROJECT {{ project.order }} · PRACTICE</p><h1>{{ project.title }}</h1><span>{{ project.description }}</span></div>
          <div class="practice-tag">单人试玩<br><small>登录后记录最佳成绩</small></div>
        </header>

        <div class="game-shell">
          <section class="game-hud">
            <div><small>{{ project.cargoTransport ? '剩余时间' : '用时' }}</small><strong class="timer">{{ project.cargoTransport ? remainingText : elapsedText }}</strong></div>
            <div><small>{{ project.cargoTransport ? '已送出' : '得分' }}</small><strong>{{ snapshot.score.toLocaleString() }}</strong></div>
            <div><small>步数</small><strong>{{ snapshot.moves }}</strong></div>
            <div v-if="project.sealEveryMoves"><small>距下次轮换</small><strong>{{ snapshot.nextSealIn }} 步</strong></div>
            <div v-if="project.resultMetric === 'boardSum' || project.targetSum"><small>盘面和</small><strong>{{ snapshot.boardSum }}</strong></div>
            <div v-if="project.targetTile"><small>{{ project.targetTile }} 数量</small><strong>{{ snapshot.targetCount }} / {{ project.targetCount }}</strong></div>
          </section>

          <div class="board-column">
            <CargoBoard v-if="project.cargoTransport" :snapshot="snapshot" :disabled="snapshot.finished" @move="move" />
            <PolyominoBoard v-else-if="project.polyomino" :snapshot="snapshot" :disabled="snapshot.finished" @move="move" />
            <TournamentBoard v-else :snapshot="snapshot" :mirror-portals="project.mirrorPortals" :irregular-shape="project.shapeShifter" :sealed-cells="snapshot.sealedCells" :disabled="snapshot.finished || locked" @move="move" />
            <div v-if="diceVisible" class="dice-curtain"><div class="die" :class="`face-${snapshot.dice}`"><i v-for="dot in 9" :key="dot"></i></div><strong>掷出 {{ snapshot.dice }} 点</strong><span>{{ dicePlacement }}</span></div>
            <div v-if="thinking" class="thinking" role="status"><span></span>AI 思考中</div>
            <div v-if="finishVisible" class="finish-panel" role="dialog" aria-label="本次试玩结果">
              <button class="finish-close" type="button" aria-label="关闭结果浮窗" @click="dismissFinish">×</button>
              <small>{{ snapshot.outcome === 'target_reached' ? 'TARGET REACHED' : snapshot.outcome === 'no_moves' ? 'NO MORE MOVES' : 'TIME LIMIT' }}</small><h2>{{ snapshot.outcome === 'target_reached' ? '完成目标' : snapshot.outcome === 'no_moves' ? '本次试玩结束' : '运输结束' }}</h2><strong>{{ project.cargoTransport ? `${snapshot.score.toLocaleString()} 块` : project.sealEveryMoves || project.polyomino ? `${snapshot.score.toLocaleString()} 分` : elapsedText }}</strong><span v-if="project.sealEveryMoves || project.polyomino || project.cargoTransport">用时 {{ elapsedText }}</span><button type="button" @click="restart">再试一次</button>
            </div>
          </div>

          <section class="game-actions">
            <button v-if="project.allowUndo" type="button" :disabled="!snapshot.canUndo || locked" @click="undo">撤销一步</button>
            <button v-if="project.allowRestart" type="button" :disabled="locked" @click="restart">重新开始</button>
            <span>方向键 / WASD / 滑动操作</span>
          </section>
        </div>
      </section>

      <aside class="rules-panel">
        <p>玩法说明</p><h2>{{ project.shortTitle }}</h2><div class="rule-copy">{{ project.description }}</div>
        <dl><dt>棋盘</dt><dd>{{ project.boardLabel || `${project.rows}×${project.cols}` }}</dd><dt>结算</dt><dd>{{ settlement }}</dd></dl>
        <div v-if="project.mirrorPortals" class="mirror-note"><strong>镜面棋盘怎么走？</strong><span>把中央十字想成真正的墙。向左滑出最左边的砖会从最右边回来；上下同理。砖最终都停在中央墙的两侧。</span><div class="mirror-mini"><i></i><i></i><i></i><i></i><b></b><em></em></div></div>
        <p v-if="snapshot.aiError" class="wasm-warning">WASM 暂不可用，本次已用确定性随机出数代替：{{ snapshot.aiError }}</p>
        <section class="practice-leaderboard" aria-label="试玩排行榜">
          <div class="leaderboard-title"><h3>试玩榜</h3><small>仅供试玩 · 不用于赛事裁决</small></div>
          <p v-if="leaderboardLoading" class="leaderboard-hint">正在加载…</p>
          <p v-else-if="leaderboardError" class="leaderboard-hint">{{ leaderboardError }}</p>
          <template v-else-if="leaderboard">
            <p v-if="!leaderboard.signed_in" class="leaderboard-hint">游客可查看；<a :href="mainSiteUrl">登录</a>后记录个人最佳。</p>
            <p v-else-if="leaderboard.my_best" class="leaderboard-mine">我的最佳：{{ formatRecord(leaderboard.my_best) }}</p>
            <p v-else class="leaderboard-hint">完成一局后记录个人最佳。</p>
            <p v-if="recordMessage" class="leaderboard-hint">{{ recordMessage }}</p>
            <ol v-if="leaderboard.top.length" class="leaderboard-list">
              <li v-for="(entry, index) in leaderboard.top" :key="entry.user_id"><span class="leaderboard-rank">{{ index + 1 }}</span><span class="leaderboard-name" :title="entry.display_name">{{ entry.display_name }}</span><strong>{{ formatRecord(entry) }}</strong></li>
            </ol>
            <p v-else class="leaderboard-hint">暂无记录，来留下第一条。</p>
          </template>
        </section>
      </aside>
    </main>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { PRACTICE_PROJECTS, PROJECT_BY_ID } from './catalog.js';
import { formatElapsed, TournamentGame } from './engine.js';
import { PolyominoGame } from './polyominoEngine.js';
import { CargoGame, CARGO_LIMIT_MS } from './cargoEngine.js';
import TournamentBoard from './TournamentBoard.vue';
import PolyominoBoard from './PolyominoBoard.vue';
import CargoBoard from './CargoBoard.vue';
import { projectIconUrl } from '../../../shared/projectIcons.js';
import { api } from '../api.js';

const props = defineProps({ projectId: { type: String, default: '' } });
const practiceThemeKey = 'tournament-practice-theme';
function initialTheme() {
  try {
    const saved = window.localStorage.getItem(practiceThemeKey);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch { /* Private browsing may disallow storage. */ }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
const practiceTheme = ref(initialTheme());
function toggleTheme() {
  practiceTheme.value = practiceTheme.value === 'dark' ? 'light' : 'dark';
  try { window.localStorage.setItem(practiceThemeKey, practiceTheme.value); } catch { /* Keep the in-page choice. */ }
}
const competitionHomePath = String(import.meta.env.VITE_COMPETITION_HOME_PATH || '/test');
const mainSiteUrl = String(import.meta.env.VITE_MAIN_SITE_URL || 'https://2048tables.online/');
const projects = PRACTICE_PROJECTS;
const project = computed(() => PROJECT_BY_ID[props.projectId] || null);
const game = ref(null);
const snapshot = ref({ board: [], score: 0, moves: 0, elapsedMs: 0 });
const now = ref(performance.now());
const locked = ref(false);
const thinking = ref(false);
const diceVisible = ref(false);
const finishVisible = ref(false);
const leaderboard = ref(null);
const leaderboardLoading = ref(false);
const leaderboardError = ref('');
const recordMessage = ref('');
let runId = 0;
let submittedRunId = -1;
let timer = null;
let diceTimer = null;
let finishTimer = null;
let thinkingTimer = null;

const elapsedText = computed(() => formatElapsed(game.value?.elapsed(now.value) || snapshot.value.elapsedMs));
const remainingText = computed(() => formatElapsed(Math.max(0, CARGO_LIMIT_MS - (game.value?.elapsed(now.value) || snapshot.value.elapsedMs))));
const dicePlacement = computed(() => snapshot.value.dice <= 3 ? '角位放置墙' : snapshot.value.dice <= 5 ? '边位放置墙' : '中心位放置墙');
const settlement = computed(() => project.value?.cargoTransport ? '无路可走或10分钟结束，按送出数量比较' : project.value?.race ? '先达到目标者获胜' : project.value?.resultMetric === 'boardSum' ? '双方死亡后比较盘面和' : '双方死亡后比较得分');

watch(() => snapshot.value.finished, finished => {
  window.clearTimeout(finishTimer);
  finishVisible.value = false;
  if (!finished) return;
  submitFinishedRun();
  finishTimer = window.setTimeout(() => {
    if (snapshot.value.finished) finishVisible.value = true;
  }, 2000);
});

function formatRecord(entry) {
  if (leaderboard.value?.metric === 'time') return formatElapsed(entry.result_value);
  return `${Number(entry.result_value).toLocaleString()}${leaderboard.value?.metric === 'deliveries' ? ' 块' : ' 分'}`;
}
async function loadLeaderboard() {
  if (!project.value) return;
  const projectId = project.value.id;
  leaderboardLoading.value = true;
  leaderboardError.value = '';
  try {
    const result = await api.practiceLeaderboard(projectId);
    if (project.value?.id === projectId) {
      leaderboard.value = result;
      if (snapshot.value.finished) submitFinishedRun();
    }
  } catch {
    if (project.value?.id === projectId) leaderboardError.value = '榜单暂不可用，不影响试玩。';
  } finally {
    if (project.value?.id === projectId) leaderboardLoading.value = false;
  }
}
async function submitFinishedRun() {
  if (!project.value || submittedRunId === runId || !leaderboard.value?.signed_in) return;
  if (project.value.race && snapshot.value.outcome !== 'target_reached') return;
  const thisRun = runId;
  const projectId = project.value.id;
  const result = {
    score: Math.trunc(snapshot.value.score || 0),
    board_sum: Math.trunc(snapshot.value.boardSum || 0),
    elapsed_ms: Math.max(1, Math.round(snapshot.value.elapsedMs || 0)),
    outcome: snapshot.value.outcome,
  };
  submittedRunId = thisRun;
  recordMessage.value = '正在记录成绩…';
  try {
    const updated = await api.submitPracticeResult(projectId, result);
    if (project.value?.id === projectId) leaderboard.value = updated;
    if (runId === thisRun) recordMessage.value = updated.improved ? '个人最佳已更新。' : '本次未超过个人最佳。';
  } catch (error) {
    if (runId === thisRun) recordMessage.value = error.status === 401 ? '登录已过期，本次未记录。' : '成绩记录失败，不影响试玩。';
  }
}

function createGame() {
  if (!project.value) return;
  runId += 1;
  game.value = project.value.cargoTransport ? new CargoGame(project.value) : project.value.polyomino ? new PolyominoGame(project.value) : new TournamentGame(project.value);
  snapshot.value = game.value.snapshot();
  if (project.value.diceWall) showDice();
}
function showDice() {
  window.clearTimeout(diceTimer);
  diceVisible.value = true;
  diceTimer = window.setTimeout(() => { diceVisible.value = false; }, 1400);
}
async function move(direction) {
  if (!game.value || locked.value || snapshot.value.finished) return;
  if (project.value.polyomino || project.value.cargoTransport) {
    snapshot.value = game.value.move(direction).snapshot;
    return;
  }
  locked.value = true;
  thinking.value = false;
  window.clearTimeout(thinkingTimer);
  if (project.value.evilSpawn) {
    thinkingTimer = window.setTimeout(() => {
      if (locked.value) thinking.value = true;
    }, 300);
  }
  try {
    const result = await game.value.move(direction);
    snapshot.value = result.snapshot;
  } finally {
    window.clearTimeout(thinkingTimer);
    thinking.value = false;
    locked.value = false;
  }
}
function restart() {
  if (!game.value || locked.value) return;
  runId += 1;
  recordMessage.value = '';
  window.clearTimeout(finishTimer);
  finishVisible.value = false;
  snapshot.value = game.value.reset(true);
  if (project.value.diceWall) showDice();
}
function undo() {
  if (locked.value) return;
  if (game.value?.undo()) snapshot.value = game.value.snapshot();
}
function dismissFinish() {
  finishVisible.value = false;
}
function keydown(event) {
  if (!project.value || locked.value || ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target?.tagName)) return;
  const direction = { ArrowUp:'up',w:'up',W:'up',ArrowRight:'right',d:'right',D:'right',ArrowDown:'down',s:'down',S:'down',ArrowLeft:'left',a:'left',A:'left' }[event.key];
  if (direction) { event.preventDefault(); move(direction); }
  if ((event.key === 'z' || event.key === 'Z') && project.value.allowUndo) { event.preventDefault(); undo(); }
}
onMounted(() => { createGame(); loadLeaderboard(); timer = window.setInterval(() => { now.value = performance.now(); if (project.value?.cargoTransport && game.value && !game.value.finished && game.value.elapsed(now.value) >= CARGO_LIMIT_MS) snapshot.value = game.value.expire(now.value); }, 16); window.addEventListener('keydown', keydown); });
onBeforeUnmount(() => { window.clearInterval(timer); window.clearTimeout(diceTimer); window.clearTimeout(finishTimer); window.clearTimeout(thinkingTimer); window.removeEventListener('keydown', keydown); });
</script>

<style scoped>
.project-art { display: block; flex: none; border-radius: 8px; object-fit: cover; }
.practice-leaderboard{margin-top:20px;padding-top:16px;border-top:1px solid #e5ded5}
.leaderboard-title h3{margin:0;font-size:17px}
.leaderboard-title small,.leaderboard-hint{color:#817469;font-size:12px;line-height:1.5}
.leaderboard-title small{display:block;margin-top:3px}
.leaderboard-hint{margin:10px 0}
.leaderboard-hint a{color:#8c6427}
.leaderboard-mine{margin:12px 0;padding:8px;border-radius:5px;background:#f0ece5;font-size:12px;font-weight:700}
.leaderboard-list{list-style:none;margin:12px 0 0;padding:0}
.leaderboard-list li{display:flex;align-items:center;gap:8px;min-width:0;padding:7px 0;border-top:1px solid #eee8df;font-size:12px}
.leaderboard-rank{width:16px;color:#9f7b3e;text-align:center;font-weight:700}
.leaderboard-name{min-width:0;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.leaderboard-list strong{font-variant-numeric:tabular-nums;white-space:nowrap}
.project-lab.is-dark .practice-leaderboard,.project-lab.is-dark .leaderboard-list li{border-color:#344354}
.project-lab.is-dark .leaderboard-title small,.project-lab.is-dark .leaderboard-hint{color:#a9b7c5}
.project-lab.is-dark .leaderboard-hint a,.project-lab.is-dark .leaderboard-rank{color:#d8ac62}
.project-lab.is-dark .leaderboard-mine{background:#263544}
.entry-art { float: left; width: 78px; height: 78px; margin: 0 16px 10px 0; }
.project-entry > small, .project-entry h2 { display: block; }
.project-entry p { clear: both; }
.rail-art { width: 25px; height: 25px; grid-row: span 2; }
.project-rail > a:not(.rail-back):has(.rail-art) { grid-template-columns: 25px 1fr; align-items: center; }
.project-rail > a:not(.rail-back):has(.rail-art) small { display: none; }
.heading-art { width: 72px; height: 72px; }
.project-heading > div { min-width: 0; flex: 1; }
@media(max-width:820px) { .heading-art { float: left; width: 58px; height: 58px; margin-right: 12px; } }
.project-lab{min-height:100vh;background:#f5f1e9;color:#332c26}.lab-header{height:66px;display:flex;align-items:center;justify-content:space-between;padding:0 max(18px,calc((100% - 1380px)/2));border-bottom:1px solid #d8d0c5;background:#fffdf8}.lab-brand{display:flex;align-items:center;gap:10px;color:inherit;text-decoration:none}.lab-brand>span{display:grid;width:38px;height:38px;place-items:center;border-radius:7px;background:#9f7b3e;color:#fff;font-weight:800}.lab-header nav{display:flex;align-items:center;gap:20px;font-size:13px}.lab-header nav a{color:#685742;text-decoration:none}.lab-header nav b{color:#928477;font-weight:400}.project-index{width:min(1120px,calc(100% - 32px));margin:auto;padding:58px 0}.index-intro p,.project-heading p,.rules-panel>p{margin:0 0 8px;color:#a36f13;font-size:12px;font-weight:800;letter-spacing:.13em}.index-intro h1{margin:0 0 12px;font-size:48px}.index-intro>span{color:#76695f}.project-list{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:34px}.project-entry{padding:22px;border:1px solid #d8d0c5;border-radius:9px;background:#fffdf8;color:inherit;text-decoration:none}.project-entry:hover{border-color:#9f7b3e}.project-entry>small{color:#a36f13;font-weight:800}.project-entry h2{margin:8px 0}.project-entry p{min-height:48px;color:#76695f}.project-entry footer{display:flex;align-items:center;gap:8px;border-top:1px solid #e6dfd6;padding-top:13px}.project-entry footer span{padding:5px 8px;background:#f0ece5;border-radius:5px;font-size:12px}.project-entry footer b{margin-left:auto;color:#8d682e;font-size:13px}.play-page{display:grid;grid-template-columns:190px minmax(540px,760px) 270px;justify-content:center;gap:24px;padding:26px 20px 50px}.project-rail{display:grid;align-content:start;gap:5px}.project-rail>a{display:grid;grid-template-columns:25px 1fr;gap:7px;padding:10px;color:#786b60;border-radius:7px;text-decoration:none;font-size:13px}.project-rail>a.active{background:#fffdf8;color:#332c26;box-shadow:inset 3px 0 #9f7b3e}.project-rail .rail-back{display:block;margin-bottom:12px;color:#8c6427}.project-heading{display:flex;justify-content:space-between;gap:20px;margin-bottom:18px}.project-heading h1{margin:0 0 7px;font-size:34px}.project-heading>div>span{color:#76695f;line-height:1.55}.practice-tag{align-self:start;flex:0 0 auto;padding:9px 12px;border:1px solid #d3c7b8;border-radius:7px;background:#fffdf8;color:#75634f;text-align:center;font-weight:700}.practice-tag small{font-weight:400}.game-shell{padding:16px;border:1px solid #d8d0c5;border-radius:10px;background:#fffdf8}.game-hud{display:flex;gap:8px;margin-bottom:14px}.game-hud>div{min-width:100px;padding:9px 12px;border-radius:7px;background:#f0ece5}.game-hud small,.game-hud strong{display:block}.game-hud small{color:#817469;font-size:11px}.game-hud strong{margin-top:2px;font-size:19px}.game-hud .timer{font-variant-numeric:tabular-nums}.board-column{position:relative;width:min(100%,620px);margin:auto}.game-actions{display:flex;align-items:center;gap:8px;margin-top:13px}.game-actions button,.finish-panel button{min-height:38px;padding:7px 12px;border:1px solid #c9bbaa;border-radius:7px;background:#fff;color:#4f4032}.game-actions button:disabled{opacity:.45}.game-actions span{margin-left:auto;color:#84776c;font-size:12px}.rules-panel{align-self:start;padding:18px;border:1px solid #d8d0c5;border-radius:9px;background:#fffdf8}.rules-panel h2{margin:0 0 10px}.rule-copy{color:#6f6258;line-height:1.6}.rules-panel dl{display:grid;grid-template-columns:75px 1fr;margin:18px 0 0}.rules-panel dt,.rules-panel dd{margin:0;padding:8px 0;border-top:1px solid #e5ded5;font-size:13px}.rules-panel dt{color:#8c8075}.mirror-note{margin-top:18px;padding:12px;border-radius:7px;background:#eee9e1}.mirror-note strong,.mirror-note span{display:block}.mirror-note span{margin-top:6px;color:#6f6258;font-size:12px;line-height:1.55}.mirror-mini{position:relative;display:grid;grid-template-columns:1fr 1fr;gap:10px;width:110px;height:110px;margin:12px auto 0}.mirror-mini i{background:#c7bcae}.mirror-mini b,.mirror-mini em{position:absolute;background:#675e55}.mirror-mini b{left:50%;width:6px;height:100%;transform:translateX(-50%)}.mirror-mini em{top:50%;width:100%;height:6px;transform:translateY(-50%)}.dice-curtain,.finish-panel{position:absolute;inset:0;z-index:20;display:grid;place-items:center;align-content:center;gap:10px;border-radius:12px;background:rgba(23,27,33,.9);color:#fff;text-align:center}.die{display:grid;grid-template:repeat(3,18px)/repeat(3,18px);gap:5px;padding:18px;border-radius:14px;background:#f7f2e8;box-shadow:0 12px 30px rgba(0,0,0,.35);animation:dice-roll .7s cubic-bezier(.2,.8,.2,1)}.die i{width:12px;height:12px;border-radius:50%;background:transparent}.face-1 i:nth-child(5),.face-2 i:nth-child(1),.face-2 i:nth-child(9),.face-3 i:nth-child(1),.face-3 i:nth-child(5),.face-3 i:nth-child(7),.face-3 i:nth-child(9),.face-4 i:nth-child(1),.face-4 i:nth-child(3),.face-4 i:nth-child(7),.face-4 i:nth-child(9),.face-5 i:nth-child(1),.face-5 i:nth-child(3),.face-5 i:nth-child(5),.face-5 i:nth-child(7),.face-5 i:nth-child(9),.face-6 i:nth-child(1),.face-6 i:nth-child(3),.face-6 i:nth-child(4),.face-6 i:nth-child(6),.face-6 i:nth-child(7),.face-6 i:nth-child(9){background:#40372f}.thinking{position:absolute;left:50%;bottom:14px;z-index:12;display:flex;align-items:center;gap:8px;transform:translateX(-50%);padding:8px 12px;border-radius:999px;background:#17212d;color:#fff;font-size:12px;white-space:nowrap;pointer-events:none;box-shadow:0 4px 14px rgba(0,0,0,.2)}.thinking span{width:12px;height:12px;border:2px solid #7f8b98;border-top-color:#e8bd65;border-radius:50%;animation:spin .7s linear infinite}.finish-panel small{color:#e7bd68;letter-spacing:.12em}.finish-panel h2{margin:0;font-size:34px}.finish-panel>strong{font-size:26px;font-variant-numeric:tabular-nums}.finish-panel button{margin-top:4px}.wasm-warning{margin-top:14px;padding:9px;background:#fff0d8;color:#795315;font-size:12px}@keyframes dice-roll{0%{transform:translateY(-80px) rotate(-240deg) scale(.5);opacity:0}70%{transform:translateY(8px) rotate(18deg) scale(1.08)}100%{transform:none;opacity:1}}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:1100px){.play-page{grid-template-columns:minmax(520px,760px) 250px}.project-rail{display:none}}@media(max-width:820px){.lab-header nav b{display:none}.play-page{display:block;padding-inline:12px}.rules-panel{margin-top:14px}.project-list{grid-template-columns:1fr}.project-heading{display:block}.practice-tag{display:inline-block;margin-top:12px}.game-hud{overflow:auto}.game-actions{flex-wrap:wrap}.game-actions span{width:100%;margin:0}.project-index{padding-top:34px}.index-intro h1{font-size:38px}}
.finish-panel{animation:finish-fade-in .3s ease both}
.face-3 i:nth-child(7){background:transparent}
.finish-panel .finish-close{position:absolute;top:12px;right:12px;display:grid;width:36px;min-height:36px;margin:0;padding:0;place-items:center;border-color:rgba(255,255,255,.3);background:rgba(255,255,255,.08);color:#fff;font-size:24px;line-height:1}
@keyframes finish-fade-in{from{opacity:0}to{opacity:1}}
.project-lab{color-scheme:light}
.theme-toggle{display:inline-flex;align-items:center;justify-content:center;gap:6px;min-height:34px;padding:5px 9px;border:1px solid #d3c7b8;border-radius:7px;background:#fffdf8;color:#685742;font-size:12px;white-space:nowrap}
.theme-toggle svg{width:16px;height:16px}.theme-toggle:hover{border-color:#9f7b3e}.theme-toggle:focus-visible{outline:2px solid #9f7b3e;outline-offset:2px}
.lab-header nav .compact-label{display:none}
.project-lab.is-dark{color-scheme:dark;background:#101821;color:#e7edf2}
.project-lab.is-dark .lab-header{background:#17212c;border-color:#344150}
.project-lab.is-dark .lab-brand>span{background:#9f7b3e}
.project-lab.is-dark .lab-header nav a,.project-lab.is-dark .rail-back{color:#c2ccd6}
.project-lab.is-dark .lab-header nav b{color:#91a0af}
.project-lab.is-dark .theme-toggle{background:#22303d;border-color:#435163;color:#e7edf2}
.project-lab.is-dark .theme-toggle:hover{border-color:#c5a05d;background:#2b3a48}
.project-lab.is-dark .index-intro p,.project-lab.is-dark .project-heading p,.project-lab.is-dark .rules-panel>p{color:#d8ac62}
.project-lab.is-dark .index-intro>span,.project-lab.is-dark .project-heading>div>span,.project-lab.is-dark .project-entry p,.project-lab.is-dark .rule-copy,.project-lab.is-dark .mirror-note span{color:#b0bdc9}
.project-lab.is-dark .project-entry,.project-lab.is-dark .game-shell,.project-lab.is-dark .rules-panel,.project-lab.is-dark .practice-tag{background:#1a2633;border-color:#344354;color:#e7edf2}
.project-lab.is-dark .project-entry:hover{background:#21303e;border-color:#b58e4e}
.project-lab.is-dark .project-entry>small,.project-lab.is-dark .project-entry footer b{color:#d8ac62}
.project-lab.is-dark .project-entry footer,.project-lab.is-dark .rules-panel dt,.project-lab.is-dark .rules-panel dd{border-color:#344354}
.project-lab.is-dark .project-entry footer span,.project-lab.is-dark .game-hud>div,.project-lab.is-dark .mirror-note{background:#263544}
.project-lab.is-dark .project-rail>a{color:#a9b7c5}
.project-lab.is-dark .project-rail>a.active{background:#263544;color:#f1f4f6;box-shadow:inset 3px 0 #c59a50}
.project-lab.is-dark .game-hud small,.project-lab.is-dark .rules-panel dt,.project-lab.is-dark .game-actions span{color:#a9b7c5}
.project-lab.is-dark .practice-tag{color:#c5d0db}
.project-lab.is-dark .game-actions button{background:#263544;border-color:#44556a;color:#e7edf2}
.project-lab.is-dark .game-actions button:hover:not(:disabled){background:#304254}
.project-lab.is-dark .mirror-mini i{background:#677585}
.project-lab.is-dark .mirror-mini b,.project-lab.is-dark .mirror-mini em{background:#263341}
.project-lab.is-dark .wasm-warning{background:#42341f;color:#f3ce8a}
.project-lab.is-dark :deep(.tournament-board),.project-lab.is-dark :deep(.tournament-board.irregular),.project-lab.is-dark :deep(.poly-board){background:#414d59}
.project-lab.is-dark :deep(.board-cell),.project-lab.is-dark :deep(.tournament-board.irregular .board-cell),.project-lab.is-dark :deep(.poly-cell){background:#64717e}
.project-lab.is-dark :deep(.mirror-cross i),.project-lab.is-dark :deep(.mirror-cross b){background:#263340;box-shadow:0 0 0 2px rgba(11,18,25,.3)}
@media(max-width:600px){.lab-header nav{gap:8px}.theme-toggle{width:34px;padding:5px}.theme-toggle span{display:none}}
@media(max-width:360px){.lab-brand strong{font-size:14px}.lab-header nav .full-label{display:none}.lab-header nav .compact-label{display:inline}}
</style>
