<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { api } from './api.js';
import { language } from './i18n.js';
import { userFacingError } from './errorMessages.js';

const props = defineProps({ slug: { type: String, required: true }, rosterRevision: Number });
const data = ref(null);
const error = ref('');
const loading = ref(false);
let disposed = false;
let timer;
let refreshQueued = false;
const number = value => value == null ? '—' : Number(value).toLocaleString('zh-CN', { maximumFractionDigits: 2 });
const date = value => new Date(typeof value === 'number' ? value * 1000 : value).toLocaleString(language.value==='en'?'en-GB':'zh-CN', { timeZone: 'Asia/Shanghai', hour12: false });
const profile = player => `https://play.2048tables.online/user/${encodeURIComponent(player.display_name)}`;
async function refresh() {
  if (loading.value) { refreshQueued = true; return; }
  loading.value = true;
  try { const next = await api.eventStatistics(props.slug); if (!disposed) { data.value = next; error.value = ''; } }
  catch (cause) { if (!disposed) error.value = userFacingError(cause); }
  finally { if (!disposed) { loading.value = false; if (refreshQueued) { refreshQueued = false; refresh(); } } }
}
watch(() => props.rosterRevision, refresh);
function refreshVisible() { if (!document.hidden) refresh(); }
onMounted(() => { refresh(); timer = setInterval(refreshVisible, 30000); document.addEventListener('visibilitychange', refreshVisible); });
onBeforeUnmount(() => { disposed = true; clearInterval(timer); document.removeEventListener('visibilitychange', refreshVisible); });
</script>

<template>
  <section class="event-statistics">
    <header class="statistics-heading"><h2>{{ $t("赛事进程与成绩") }}</h2><button type="button" :disabled="loading" @click="refresh">{{ $t(loading ? '更新中…' : '刷新成绩') }}</button></header>
    <p v-if="error" class="alert" role="alert">{{ $t(error) }}<span v-if="data">{{ $t(" 以下为上次成功读取的数据。") }}</span></p>
    <template v-if="data">
      <p class="statistics-meta">{{ $t({ upcoming: '尚未开赛', active: '比赛进行中', ended: '统计时间窗已结束' }[data.phase]) }}{{ $t(" · 更新于 ") }}{{ $t(date(data.as_of)) }}{{ $t("（北京时间） · 页面可见时每 30 秒刷新") }}</p>
      <p class="statistics-meta">{{ $t(data.rating_note) }}</p>
      <p v-if="!data.players.length" class="statistics-empty">{{ $t("等待报名或举办方导入名单。目前尚未分组，不展示虚构队伍或成绩。") }}</p>
      <p v-if="data.unassigned_count" class="statistics-meta">{{ $t(data.unassigned_count) }}{{ $t(" 位选手待分组，个人成绩照常统计；团队成绩在分组后汇总。") }}</p>
      <div class="team-statistics">
        <article v-for="team in data.teams" :key="team.name" class="team-stat-card">
          <header><h3>{{ team.name }}</h3><span>{{ $t(team.complete ? '全员满五局' : '成绩未满') }}</span></header>
          <div class="team-metrics"><div><small>{{ $t("盘面和合计") }}</small><strong>{{ $t(number(team.board_sum)) }}</strong></div><div><small>{{ $t("Rating 合计") }}</small><strong>{{ $t(number(team.rating)) }}</strong></div></div>
          <progress :value="team.selected_games" max="25" :aria-label="`${team.name}已入选${team.selected_games}局，共需25局`" />
          <p class="statistics-meta">{{ $t("已入选 ") }}{{ $t(team.selected_games) }}{{ $t("/25 局 · 有效完赛 ") }}{{ $t(team.completed_games) }}{{ $t(" 局") }}</p>
          <div v-for="player in team.players" :key="player.user_id" class="team-member"><a :href="profile(player)" target="_blank" rel="noopener noreferrer">{{ player.display_name }}<small v-if="player.is_external">{{ $t(" · 外援") }}</small></a><span>{{ $t(player.selected_games) }}{{ $t("/5 局") }}</span></div>
        </article>
      </div>
      <section v-if="data.players.length" class="player-statistics"><h2>{{ $t("个人成绩与最佳五局") }}</h2><p class="statistics-meta">{{ $t("按个人 rating 排列。分数决定入选局；不足五局按零补足后计算。* 表示尚未完成五局。") }}</p>
        <details v-for="player in data.players" :key="player.user_id" class="player-stat-row"><summary><strong>{{ player.display_name }}{{ $t(player.is_external ? ' · 外援' : '') }}</strong><span>{{ player.team_name || '待分组' }}</span><span>{{ $t(player.selected_games) }}{{ $t("/5 局") }}</span><span>{{ $t("盘面和 ") }}{{ $t(number(player.board_sum)) }}</span><span>Rating {{ $t(number(player.rating)) }}{{ $t(player.complete ? '' : ' *') }}</span></summary>
          <div class="games-scroll"><table><thead><tr><th>{{ $t("得分") }}</th><th>{{ $t("盘面和") }}</th><th>{{ $t("开始时间（北京时间）") }}</th><th>{{ $t("完成时间") }}</th></tr></thead><tbody><tr v-for="game in player.games" :key="game.id"><td>{{ $t(number(game.score)) }}</td><td>{{ $t(number(game.board_sum)) }}</td><td>{{ $t(date(game.started_at)) }}</td><td>{{ $t(date(game.ended_at)) }}</td></tr></tbody></table></div><p v-if="!player.games.length">{{ $t("暂无符合时间和有效性要求的已完成局。") }}</p>
        </details>
      </section>
    </template>
  </section>
</template>

<style scoped>
.event-statistics{margin:24px 0}.statistics-heading,.team-stat-card header{display:flex;justify-content:space-between;align-items:center;gap:16px}.statistics-heading h2{margin:0}.statistics-meta{font-size:13px;line-height:1.8;color:#817567}.team-statistics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.team-stat-card,.roster-import{border:1px solid #e3d8c9;border-radius:8px;padding:20px;background:#fffdf8}.team-stat-card h3{margin:0;font-size:22px}.team-stat-card header>span{font-size:12px;color:#817567}.team-metrics{display:flex;gap:30px;margin:24px 0}.team-metrics div{display:grid;gap:8px}.team-metrics small{color:#817567}.team-metrics strong{font-size:24px}progress{width:100%;height:8px;accent-color:#a18145}.team-member{display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-bottom:1px solid #eee7db}.team-member a{color:inherit;text-decoration:none}.team-member span{font-size:13px;color:#817567}.player-statistics{margin:32px 0}.player-stat-row{border-bottom:1px solid #e3d8c9;padding:16px 0}.player-stat-row summary{display:flex;flex-wrap:wrap;gap:12px 22px;cursor:pointer;font-size:14px}.player-stat-row summary strong{min-width:120px}.player-stat-row summary::before{content:'▸';color:#a18145}.player-stat-row[open] summary::before{content:'▾'}.games-scroll{overflow-x:auto;margin-top:16px}table{width:100%;border-collapse:collapse;font-size:13px;text-align:left;white-space:nowrap}td,th{padding:12px 10px;border-bottom:1px solid #eee7db}.roster-import{margin:24px 0}.roster-import summary{cursor:pointer}.roster-import p{line-height:1.7}.roster-import label{display:grid;gap:10px;margin:20px 0}textarea{box-sizing:border-box;width:100%;padding:12px;border:1px solid #cdd2d8;border-radius:5px;font:inherit;color:inherit;background:#f5f6f8}button{cursor:pointer;padding:9px 15px;border:1px solid #cbb895;border-radius:5px;background:#fffdf8;color:inherit}button:disabled{opacity:.5;cursor:default}.roster-preview{margin-top:20px}.statistics-empty{padding:30px;border:1px dashed #cbb895;border-radius:8px}.primary-button{background:#a18145;color:white}@media(max-width:650px){.team-statistics{grid-template-columns:1fr}.team-stat-card{padding:16px}.team-metrics{gap:20px}.team-metrics strong{font-size:22px}.statistics-heading h2{font-size:20px}.player-stat-row summary strong{width:calc(100% - 40px)}}
</style>
