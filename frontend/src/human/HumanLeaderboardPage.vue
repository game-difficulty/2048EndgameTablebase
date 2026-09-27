<template>
  <section class="full-leaderboard-page" :aria-busy="loading">
    <header class="full-leaderboard-title">
      <div><span>{{ L('正式成绩','Ranked results') }}</span><h1>{{ L('排行榜','Leaderboards') }}</h1></div>
      <a href="/#game" @click.prevent="$emit('back')">{{ L('返回对局','Back to game') }}</a>
    </header>

    <nav class="leaderboard-kinds" :aria-label="L('榜单类型','Leaderboard type')">
      <button v-for="item in kinds" :key="item.id" :class="{ active: filters.type === item.id }" @click="setType(item.id)">{{ item.label }}</button>
    </nav>

    <div class="leaderboard-filters">
      <div v-if="filters.type !== 'rate32k' && filters.type !== 'strength'" class="leaderboard-filter-group">
        <span>{{ L('棋盘','Board') }}</span>
        <button v-for="item in variants" :key="item" :class="{ active: filters.variant === item }" @click="setFilter('variant',item)">{{ item.replace('x',' × ') }}</button>
      </div>
      <div v-if="filters.type === 'score' || filters.type === 'strength'" class="leaderboard-filter-group">
        <span>{{ L('周期','Period') }}</span>
        <button :class="{ active: filters.period === 'all' }" @click="setFilter('period','all')">{{ L('总榜','All time') }}</button>
        <button :class="{ active: filters.period === 'week' }" @click="setFilter('period','week')">{{ L('近7天','Last 7 days') }}</button>
      </div>
      <label v-if="filters.type === 'strength'" class="leaderboard-pattern-select">
        <span>{{ L('定式','Formation') }}</span>
        <select :value="formationKey" :disabled="!formations.length" @change="setFormation($event.target.value)">
          <option v-if="!formations.length" value=":">{{ L('暂无可用定式','No available formations') }}</option>
          <option v-for="item in formations" :key="`${item.pattern}:${item.target}`" :value="`${item.pattern}:${item.target}`">{{ item.pattern }}-{{ item.target }}</option>
        </select>
      </label>
    </div>

    <p v-if="filters.type === 'rate32k'" class="leaderboard-explanation">
      {{ L('仅统计至少有 10 局达到 32K 的玩家；分母为有回放覆盖的 32K 残局阶段。','Players need at least ten 32K games. The denominator includes 32K endgame stages covered by replays.') }}
    </p>
    <p v-else-if="filters.type === 'strength'" class="leaderboard-explanation">
      {{ L('仅计入从个人主页历史对局发起的“对局－定式”分析；每位玩家仅取该定式的最佳分析，同等级按内部综合结果排序。','Only game–formation analyses started from Profile history are eligible. Each player contributes their best result for the formation; equal grades are ordered by the private combined result.') }}
    </p>

    <div class="full-leaderboard-panel">
      <div class="full-leaderboard-heading">
        <span>{{ L('排名','Rank') }}</span><span>{{ L('玩家','Player') }}</span><span>{{ metricTitle }}</span><span>{{ L('详情','Details') }}</span>
      </div>
      <div v-if="loading" class="large-empty">{{ L('正在读取榜单…','Loading leaderboard…') }}</div>
      <div v-else-if="error" class="large-empty"><p>{{ L('榜单暂不可用','Leaderboard unavailable') }}</p><button @click="load(true)">{{ L('重试','Retry') }}</button></div>
      <div v-else-if="!data.entries?.length" class="large-empty">{{ L('尚无符合条件的成绩','No eligible results yet') }}</div>
      <div v-else class="full-leaderboard-rows">
        <article v-for="item in data.entries" :key="`${filters.type}:${item.user_id}`" class="full-leaderboard-row">
          <strong class="full-rank notranslate" :class="{ podium: item.rank <= 3 }" translate="no">{{ item.rank }}</strong>
          <button class="full-player notranslate" translate="no" :title="item.display_name" @click="$emit('player',item.display_name)">{{ item.display_name }}</button>
          <button v-if="filters.type === 'score'" class="full-metric metric-link" :disabled="!item.has_replay" @click="$emit('replay',replayItem(item))">{{ integer(item.score) }}</button>
          <strong v-else-if="filters.type === 'rating'" class="full-metric">{{ decimal(item.rating) }}</strong>
          <strong v-else-if="filters.type === 'rate32k'" class="full-metric">{{ percent(item.rate_32k_value) }}</strong>
          <strong v-else-if="filters.type === 'count'" class="full-metric">{{ integer(item.primary_achievement_count) }}</strong>
          <strong v-else class="full-metric strength-grade" :class="`grade-${String(item.grade).toLowerCase()}`">{{ item.grade }}</strong>
          <div class="full-details">
            <template v-if="filters.type === 'score'"><span>{{ date(item.ended_at) }}</span></template>
            <template v-else-if="filters.type === 'rating'"><span>{{ L('当前 B10 线','Current B10 line') }} · {{ item.b10_score == null ? '—' : integer(item.b10_score) }}</span></template>
            <template v-else-if="filters.type === 'rate32k'"><span>{{ integer(item.rate_32k_passed) }} / {{ integer(item.rate_32k_total) }} {{ L('阶段','stages') }}</span><small>{{ L(`${item.rate_32k_candidate_games} 局达到 32K`,`${item.rate_32k_candidate_games} games reached 32K`) }}</small></template>
            <template v-else-if="filters.type === 'count'"><span>{{ L(`共 ${item.game_count} 局正式记录`,`${item.game_count} ranked games`) }}</span></template>
            <template v-else><span>{{ percent(item.mean_goodness_of_fit) }} · {{ item.stage_count }} {{ L('个残局','stages') }} · {{ integer(item.evaluated_moves) }} {{ L('评价步','rated moves') }}</span><button class="strength-run" @click="$emit('replay',replayItem(item))">{{ integer(item.final_score) }} · {{ date(item.run_ended_at) }}</button></template>
          </div>
        </article>
      </div>
    </div>

    <nav v-if="data.page_count > 1" class="full-leaderboard-pagination" :aria-label="L('排行榜分页','Leaderboard pages')">
      <button :disabled="data.page <= 1" @click="setPage(1)">«</button>
      <button :disabled="data.page <= 1" @click="setPage(data.page - 1)">‹</button>
      <span>{{ L(`第 ${data.page} / ${data.page_count} 页 · ${data.total} 人`,`Page ${data.page} of ${data.page_count} · ${data.total} players`) }}</span>
      <button :disabled="data.page >= data.page_count" @click="setPage(data.page + 1)">›</button>
      <button :disabled="data.page >= data.page_count" @click="setPage(data.page_count)">»</button>
    </nav>
  </section>
</template>

<script setup>
import { computed, onMounted, onUnmounted, reactive, ref } from 'vue';
import { json } from './client.js';
import { language } from './i18n.js';

defineEmits(['back','player','replay']);
const variants = ['4x4','3x4','3x3','2x4'];
const L = (zh,en) => language.value === 'en' ? en : zh;
const kinds = computed(() => [
  { id:'score', label:L('分数榜','Score') }, { id:'rating', label:'Rating' },
  { id:'rate32k', label:L('综率榜','32K rate') }, { id:'count', label:L('数量榜','Milestones') },
  { id:'strength', label:L('残局实力榜','Endgame skill') },
]);
const filters = reactive({ type:'score', variant:'4x4', period:'all', pattern:'', target:'', page:1 });
const catalog = ref({ strength:[] }), data = ref({ entries:[], page:1, page_count:1, total:0 });
const loading = ref(false), error = ref(''); let serial = 0;
const formations = computed(() => catalog.value.strength?.filter(item => item.variant === '4x4') || []);
const formationKey = computed(() => `${filters.pattern}:${filters.target}`);
const metricTitle = computed(() => ({ score:L('分数','Score'), rating:'B10 Rating', rate32k:L('32K 综率','32K rate'), count:`${data.value.achievement || ({'4x4':'32K','3x4':'4096','3x3':'1024','2x4':'512'}[filters.variant])} ${L('数量','count')}`, strength:L('等级','Grade') })[filters.type]);
const integer = value => Number(value || 0).toLocaleString(language.value === 'en' ? 'en-US' : 'zh-CN');
const decimal = value => value == null ? '—' : Number(value).toLocaleString(undefined,{minimumFractionDigits:1,maximumFractionDigits:1});
const percent = value => value == null ? '—' : `${(Number(value) * 100).toFixed(2)}%`;
const date = value => value == null ? '—' : new Date(Number(value) * 1000).toLocaleString(language.value === 'en' ? 'en-US' : 'zh-CN',{year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});
const replayItem = item => ({ id:item.run_id, source:item.source || 'native', has_replay:item.has_replay !== false });

function normalize() {
  const query = new URLSearchParams(location.search), requestedType = query.get('type');
  filters.type = ['score','rating','rate32k','count','strength'].includes(requestedType) ? requestedType : 'score';
  const requestedVariant = query.get('variant'); filters.variant = variants.includes(requestedVariant) ? requestedVariant : '4x4';
  if (filters.type === 'rate32k' || filters.type === 'strength') filters.variant = '4x4';
  filters.period = ['score','strength'].includes(filters.type) && query.get('period') === 'week' ? 'week' : 'all';
  filters.pattern = query.get('pattern') || ''; filters.target = query.get('target') || '';
  filters.page = Math.max(1, Math.min(100000, Number.parseInt(query.get('page') || '1',10) || 1));
}
function url() {
  const query = new URLSearchParams({ type:filters.type, variant:filters.variant, period:filters.period, page:String(filters.page) });
  if (filters.type === 'strength') { query.set('pattern',filters.pattern); query.set('target',filters.target); }
  return `/leaderboard?${query}`;
}
function navigate(replace=false) { const next=url(); (replace ? history.replaceState : history.pushState).call(history,null,'',next); void load(); }
function setType(type) { filters.type=type; filters.page=1; if (type === 'rate32k' || type === 'strength') filters.variant='4x4'; filters.period=['score','strength'].includes(type)?filters.period:'all'; ensureFormation(); navigate(); }
function setFilter(key,value) { filters[key]=value; filters.page=1; navigate(); }
function setFormation(value) { const split=value.lastIndexOf(':'); filters.pattern=value.slice(0,split); filters.target=value.slice(split+1); filters.page=1; navigate(); }
function setPage(page) { filters.page=page; navigate(); }
function ensureFormation() {
  if (filters.type !== 'strength') return;
  const match=formations.value.find(item => item.pattern===filters.pattern && String(item.target)===String(filters.target));
  if (!match && formations.value.length) { filters.pattern=formations.value[0].pattern; filters.target=String(formations.value[0].target); }
}
async function load(force=false) {
  ensureFormation();
  if (filters.type === 'strength' && !filters.pattern) { data.value={entries:[],page:1,page_count:1,total:0}; return; }
  const own=++serial; loading.value=true; error.value='';
  const query=new URLSearchParams({ type:filters.type,variant:filters.variant,period:filters.period,page:String(filters.page),page_size:'50' });
  if (filters.type==='strength') { query.set('pattern',filters.pattern); query.set('target',filters.target); }
  try { const result=await json(`/api/human/leaderboards/full?${query}`); if (own===serial) data.value=result; }
  catch { if (own===serial) error.value='leaderboard_unavailable'; }
  finally { if (own===serial) loading.value=false; }
}
async function start() {
  normalize();
  try { catalog.value=await json('/api/human/leaderboards/catalog'); } catch { catalog.value={strength:[]}; }
  ensureFormation(); history.replaceState(null,'',url()); await load();
}
function popstate() { normalize(); ensureFormation(); void load(); }
onMounted(() => { window.addEventListener('popstate',popstate); void start(); });
onUnmounted(() => window.removeEventListener('popstate',popstate));
</script>
