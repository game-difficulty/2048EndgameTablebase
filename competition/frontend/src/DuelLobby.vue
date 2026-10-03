<script setup>
import { computed, onMounted, ref } from 'vue';
import { api } from './api.js';
import { language, t } from './i18n.js';
import { PROJECT_BY_ID } from './projects/catalog.js';
import { projectIconUrl } from '../../shared/projectIcons.js';
import { userFacingError } from './errorMessages.js';

defineProps({ session: Object, rooms: { type: Array, default: () => [] }, mainSiteUrl: String });
const emit = defineEmits(['navigate']);
const en = computed(() => language.value === 'en');
const catalog = ref([]), selected = ref([]), name = ref(''), minutes = ref(30), code = ref('');
const busy = ref(false), error = ref('');
let commandId = crypto.randomUUID();
function resetCommand() { commandId = crypto.randomUUID(); }
function title(project) { return t(PROJECT_BY_ID[project.project_ref]?.shortTitle || project.display_name); }
function toggle(project) {
  const index = selected.value.indexOf(project);
  if (index >= 0) selected.value.splice(index, 1);
  else if (selected.value.length < 15) selected.value.push(project);
  commandId = crypto.randomUUID();
}
function move(index, offset) {
  const target = index + offset;
  if (target < 0 || target >= selected.value.length) return;
  [selected.value[index], selected.value[target]] = [selected.value[target], selected.value[index]];
  commandId = crypto.randomUUID();
}
async function loadCatalog() {
  try { catalog.value = (await api.duelProjects()).projects; }
  catch (cause) { error.value = userFacingError(cause); }
}
async function create() {
  busy.value = true; error.value = '';
  try {
    const payload = await api.createDuel({ name: name.value.trim() || (en.value ? 'Free duel' : '自由对决'),
      projects: selected.value.map(p => p.project_ref), clock_seconds: Math.round(Number(minutes.value) * 60), command_id: commandId });
    emit('navigate', `/rooms/${payload.competition.room_code}`);
  } catch (cause) { error.value = userFacingError(cause); }
  finally { busy.value = false; }
}
onMounted(loadCatalog);
</script>

<template>
  <main class="page-width duel-lobby">
    <header class="duel-intro">
      <div><p class="eyebrow">FREE DUEL · 1 VS 1</p><h1>{{ en ? 'Your projects. Your match.' : '选好项目，来一场对决。' }}</h1>
      <p>{{ en ? 'Invite a friend, ready up together, and play every selected project in order.' : '邀请一位对手，双方准备后，依次完成所选项目。' }}</p></div>
      <form class="join-box" @submit.prevent="code.trim() && emit('navigate', `/rooms/${encodeURIComponent(code.trim().toUpperCase())}`)">
        <label for="duel-code">{{ en ? 'Room code' : '房间码' }}</label>
        <div class="inline-form"><input id="duel-code" v-model="code" maxlength="12" required /><button class="primary-button">{{ en ? 'Join' : '进入' }}</button></div>
      </form>
    </header>
    <p v-if="error" class="alert" role="alert">{{ t(error) }}</p>
    <p v-if="!session" class="panel">{{ en ? 'Sign in to create or join a duel.' : '登录后即可创建或加入自由对决。' }} <a :href="mainSiteUrl">{{ en ? 'Sign in' : '前往登录' }}</a></p>
    <form v-else class="duel-builder" @submit.prevent="create">
      <section class="panel">
        <h2>{{ en ? 'Choose projects' : '选择项目' }} <small>{{ selected.length }} / 15</small></h2>
        <p class="muted">{{ en ? 'Only registered projects are offered. Each project can be selected once.' : '仅使用服务端已注册项目，每个项目最多选择一次。' }}</p>
        <button v-if="!catalog.length" type="button" class="secondary-button" @click="loadCatalog">{{ en ? 'Reload projects' : '重新加载项目' }}</button>
        <div class="duel-project-grid">
          <button v-for="project in catalog" :key="project.project_ref" type="button" :aria-pressed="selected.includes(project)" :disabled="busy || (!selected.includes(project) && selected.length >= 15)" @click="toggle(project)">
            <img :src="projectIconUrl(project.project_ref)" alt="" /><span>{{ title(project) }}</span><b>{{ selected.includes(project) ? '✓' : '+' }}</b>
          </button>
        </div>
      </section>
      <section class="panel duel-order">
        <h2>{{ en ? 'Match order' : '对战顺序' }}</h2>
        <p v-if="!selected.length" class="muted">{{ en ? 'Select a project to begin.' : '先选择一个项目。' }}</p>
        <ol><li v-for="(project,index) in selected" :key="project.project_ref"><span>{{ index+1 }}. {{ title(project) }}</span>
          <button type="button" :disabled="busy || index===0" :aria-label="en ? 'Move up' : '上移'" @click="move(index,-1)">↑</button>
          <button type="button" :disabled="busy || index===selected.length-1" :aria-label="en ? 'Move down' : '下移'" @click="move(index,1)">↓</button>
          <button type="button" :disabled="busy" :aria-label="en ? 'Remove project' : '移除项目'" @click="toggle(project)">×</button>
        </li></ol>
        <label>{{ en ? 'Room name' : '房间名称' }}<input v-model="name" maxlength="100" minlength="2" :disabled="busy" :placeholder="en ? 'Free duel' : '自由对决'" @input="resetCommand" /></label>
        <label>{{ en ? 'Total time per player (minutes)' : '每人总用时（分钟）' }}<input v-model="minutes" type="number" min="1" max="1440" step="1" required :disabled="busy" @input="resetCommand" /></label>
        <p class="muted">{{ en ? 'Rules and order are fixed on creation. No BP, referees, or automatic ready-up. All projects share the time budget; the existing Higher time-refund rule applies.' : '创建后规则与顺序固定。无 BP、无裁判、不自动准备。全部项目共用总用时，沿用 Higher 项目的差额补时规则。' }}</p>
        <p class="muted">{{ en ? 'One active duel per user. Unstarted rooms and between-game waits expire after 30 minutes. Creation: once per minute, up to 10 per hour.' : '每人最多一个进行中的自由对决。开赛前或局间等待满 30 分钟关闭。每分钟最多创建一次，每小时最多 10 次。' }}</p>
        <button class="primary-button" :disabled="busy || !selected.length">{{ busy ? (en ? 'Creating…' : '创建中…') : (en ? 'Create duel' : '创建自由对决') }}</button>
      </section>
    </form>
    <section class="panel duel-history"><h2>{{ en ? 'My duels' : '我的自由对决' }}</h2>
      <p v-if="!rooms.length" class="muted">{{ en ? 'No duels yet.' : '暂无自由对决。' }}</p>
      <button v-for="item in rooms" :key="item.id" class="room-row room-row-link" @click="emit('navigate', `/rooms/${item.room_code}`)"><span><strong>{{ item.name }}</strong><small>{{ item.room_code }} · {{ item.rules.game_count }} {{ en ? 'games' : '局' }}</small></span><span>{{ item.series_score?.yellow ?? 0 }} : {{ item.series_score?.white ?? 0 }} · {{ ['FINISHED','CANCELLED'].includes(item.status) ? (en ? 'Ended' : '已结束') : (en ? 'In progress' : '进行中') }}</span></button>
    </section>
  </main>
</template>

<style scoped>
.duel-builder>.panel,.duel-history{padding:24px;min-width:0}.duel-order li span{min-width:0;overflow-wrap:anywhere}.duel-order li button{border:1px solid var(--competition-border);border-radius:6px;background:var(--competition-card);color:inherit}
.duel-lobby{padding-top:32px;padding-bottom:48px}.duel-intro{display:flex;justify-content:space-between;gap:24px;align-items:center;margin-bottom:24px}.duel-intro h1{font-size:clamp(24px,3vw,36px);margin:8px 0}.duel-intro p{max-width:640px;line-height:1.6}.duel-builder{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr);gap:24px}.duel-builder h2{margin-top:0}.duel-builder small{font-size:14px;opacity:.6}.duel-project-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.duel-project-grid button{display:flex;align-items:center;gap:10px;min-width:0;text-align:left;padding:12px;border:1px solid var(--line,#d9d3c8);border-radius:12px;background:transparent;color:inherit}.duel-project-grid button[aria-pressed=true]{border-color:#b88b37;background:#b88b3715}.duel-project-grid img{width:38px;height:38px}.duel-project-grid span{flex:1;overflow-wrap:anywhere}.duel-order ol{padding:0;list-style:none}.duel-order li{display:flex;gap:6px;align-items:center;padding:10px 0;border-bottom:1px solid #8883}.duel-order li span{flex:1}.duel-order li button{min-width:34px;min-height:34px}.duel-order label{display:grid;gap:6px;margin:18px 0}.duel-order input{width:100%;box-sizing:border-box}.duel-order .muted{font-size:13px;line-height:1.7}.duel-order>.primary-button{width:100%}.duel-history{margin-top:24px}button:disabled{opacity:.5}
@media(max-width:760px){.duel-intro{display:block}.duel-intro .join-box{margin-top:20px}.duel-builder{grid-template-columns:1fr}.duel-project-grid button{padding:10px;font-size:14px}.duel-project-grid img{width:30px;height:30px}.duel-builder>.panel,.duel-history{padding:16px}.duel-order li button{min-width:40px;min-height:44px}}
</style>
