<script setup>
import { computed, ref, watch, onMounted, onBeforeUnmount } from 'vue';
import { scheduleTime, roomMatchTitle, roomMatchScore } from './scheduleDisplay.js';
import { api } from './api.js';
import { userFacingError } from './errorMessages.js';
import { language } from './i18n.js';
const txt=(zh,en)=>language.value==='zh'?zh:en;
const tab=ref('overview');
const tabs=computed(()=>[['overview',txt('赛事概览','Overview')],['teams',txt('报名与队伍','Registration & teams')],['results',txt('赛程与成绩','Schedule & results')],...(detail.value?.can_manage || detail.value?.can_assign_organizer ? [['manage',txt('赛事管理','Manage event')]]:[])]);
const activeCount=computed(()=>props.events.filter(e=>e.status==='active').length);
const unboundRooms=computed(()=>detail.value?.rooms?.filter(r=>!r.fixture_id)||[]);
import EventStatistics from './EventStatistics.vue';
import EventEnrollment from './EventEnrollment.vue';
import EventFixtures from './EventFixtures.vue';

const props = defineProps({ slug: { type: String, default: '' }, events: { type: Array, default: () => [] }, canCreate: Boolean, statusText: Object });
const emit = defineEmits(['navigate', 'refresh', 'create-room']);
const detail = ref(null);
const loading = ref(false);
const error = ref('');
const busy = ref(false);
const filter = ref('all');
const roomCode = ref('');
const draft = ref({ slug: '', name: '', description: '', rules: '', team_size: 3 });
const settings = ref({});
const notice = ref('');
const enrollmentRevision = ref(0);
const organizerId = ref(''), organizerPreview = ref(null);
async function assignOrganizer(save = false) {
  if (busy.value) return;
  const userId = Number(organizerId.value), slug = props.slug;
  if (!Number.isSafeInteger(userId) || userId <= 0) { error.value = '请输入有效的 Table 用户 ID。'; return; }
  if (save && organizerPreview.value?.user_id !== userId) return;
  busy.value = true; error.value = ''; notice.value = '';
  try {
    const result = await api.assignEventOrganizer(slug, { user_id:userId, dry_run:!save });
    if (props.slug !== slug) return;
    if (save) { detail.value = result.event; organizerPreview.value = null; notice.value = '举办方已更新。'; }
    else organizerPreview.value = result.organizer;
  } catch(cause) { if(props.slug === slug) error.value = userFacingError(cause); }
  finally { busy.value = false; }
}
const labels = { preparing: '筹备中', active: '进行中', finished: '已结束' };
const shown = computed(() => props.events.filter(event => filter.value === 'all' || event.status === filter.value));
let epoch = 0;
let refreshTimer;
onMounted(() => { refreshTimer = setInterval(async () => {
  if (!props.slug || document.hidden || busy.value) return;
  const slug = props.slug;
  try { const payload = await api.event(slug); if(props.slug === slug) detail.value = payload.event; } catch { /* Next refresh retries. */ }
}, 15000); });
onBeforeUnmount(() => clearInterval(refreshTimer));
watch(() => props.slug, async slug => {
  const id = ++epoch;
  tab.value = 'overview';
  error.value = ''; detail.value = null; roomCode.value = ''; loading.value = !!slug;
  organizerId.value = ''; organizerPreview.value = null;
  if (!slug) return;
  try {
    const payload = await api.event(slug);
    if (id === epoch) {
      detail.value = payload.event;
      settings.value = { name: payload.event.name, description: payload.event.description, rules: payload.event.rules, status: payload.event.status };
    }
  } catch (cause) { if (id === epoch) error.value = userFacingError(cause); }
  finally { if (id === epoch) loading.value = false; }
}, { immediate: true });
async function create() {
  if (busy.value) return;
  busy.value = true; error.value = '';
  try {
    const result = await api.createEvent(draft.value);
    emit('navigate', `/events/${result.event.slug}`);
  } catch (cause) { error.value = userFacingError(cause); }
  finally { busy.value = false; }
}
async function linkRoom() {
  if (busy.value) return;
  busy.value = true; error.value = '';
  const slug = props.slug;
  try {
    const result = await api.linkEventRoom(slug, roomCode.value.trim().toUpperCase());
    if (props.slug === slug) { detail.value = result.event; roomCode.value = ''; }
    emit('refresh');
  } catch (cause) { if (props.slug === slug) error.value = userFacingError(cause); }
  finally { busy.value = false; }
}
async function save() {
  if (busy.value) return;
  busy.value = true; error.value = ''; notice.value = '';
  const slug = props.slug;
  try {
    const result = await api.updateEvent(slug, settings.value);
    if (props.slug === slug) { detail.value = result.event; notice.value = '赛事信息已保存。'; }
    emit('refresh');
  } catch (cause) { if (props.slug === slug) error.value = userFacingError(cause); }
  finally { busy.value = false; }
}
</script>

<template>
  <section class="event-center">
    <p v-if="error" class="alert" role="alert">{{ $t(error) }}</p>
    <p v-if="notice" role="status">{{ $t(notice) }}</p>
    <template v-if="!slug">
      <div class="event-intro"><div><p class="eyebrow">2048 · COMPETITION</p><h1>{{ txt("在这里，下一场精彩开始。","Your next great match starts here.") }}</h1><p class="muted">{{ txt("发现赛事、集结队伍、关注每一场对决。","Discover events, bring your team, and follow every match.") }}</p></div><a href="/practice">{{ $t("项目练习 →") }}</a></div>
      <div class="lobby-stats"><div><strong>{{ events.length }}</strong><span>{{ txt("全部赛事","Events") }}</span></div><div><strong>{{ activeCount }}</strong><span>{{ txt("正在进行","Live now") }}</span></div><div><strong>{{ events.reduce((n,e)=>n+(e.room_count||0),0) }}</strong><span>{{ txt("比赛房间","Match rooms") }}</span></div></div>
      <nav class="event-filters" :aria-label='$t("赛事状态")'><button v-for="(label, key) in { all: '全部赛事', ...labels }" :key="key" :aria-pressed="filter === key" @click="filter = key">{{ $t(label) }}</button></nav>
      <div class="event-cards">
        <a v-for="event in shown" :key="event.slug" :href="`/events/${event.slug}`" :class="['event-card',event.status]" @click.prevent="emit('navigate', `/events/${event.slug}`)">
          <div class="event-cover"><span>2048 · TOURNAMENT</span><strong>{{ $t(event.name) }}</strong><small>{{ event.capabilities.rooms ? txt(`${event.team_size || 3} 人团队赛`,`${event.team_size || 3}-player teams`) : $t(event.label) }}</small></div>
          <div class="event-card-body"><span class="event-status"><i></i>{{ $t(labels[event.status]) }}</span><p>{{ $t(event.description || '进入赛事查看详情。') }}</p><footer><span>{{ $t(event.capabilities.statistics ? '名单导入 · 成绩统计' : `${event.room_count} 个比赛房间`) }}</span><span>{{ $t("查看赛事 →") }}</span></footer></div>
        </a>
      </div>
      <p v-if="!shown.length" class="empty-state">{{ $t("暂无此状态的赛事。") }}</p>
      <details v-if="canCreate" class="event-admin"><summary>{{ $t("创建赛事") }}</summary><form @submit.prevent="create">
        <label>{{ $t("赛事名称") }}<input v-model="draft.name" required minlength="2" maxlength="100" /></label>
        <label>{{ txt("每队人数（创建后固定）","Players per team (fixed on creation)") }}<input v-model.number="draft.team_size" type="number" min="1" max="16" required /></label>
        <label>{{ $t("赛事地址标识") }}<input v-model="draft.slug" required minlength="2" maxlength="64" pattern="[a-z0-9]+(-[a-z0-9]+)*" :placeholder='$t("例如 summer-cup-1")' /></label>
        <label>{{ $t("简介") }}<textarea v-model="draft.description" maxlength="1000" /></label>
        <label>{{ $t("规则与公告") }}<textarea v-model="draft.rules" maxlength="10000" rows="4" /></label>
        <p class="muted">{{ txt("创建团队选 Ban 赛事。默认关闭报名；比赛房间的局数和 BP 流程可分别配置。","Creates a team draft event with registration closed. Configure game count and BP rules separately in each room.") }}</p>
        <button class="primary-button" :disabled="busy">{{ $t("创建赛事") }}</button>
      </form></details>
    </template>
    <template v-else>
      <a href="/events" @click.prevent="emit('navigate', '/events')">{{ $t("← 全部赛事") }}</a>
      <p v-if="loading" class="empty-state">{{ $t("正在加载赛事…") }}</p>
      <template v-else-if="detail">
        <header class="event-detail-title"><p class="eyebrow">2048 · TOURNAMENT</p><span class="event-status">{{ $t(labels[detail.status]) }} · {{ detail.capabilities.rooms?txt(`${detail.team_size} 人团队赛`,`${detail.team_size}-player teams`):$t(detail.label) }}</span><h1>{{ $t(detail.name) }}</h1><p class="muted">{{ $t(detail.description) }}</p><div class="detail-actions"><button type="button" class="primary-button" @click="tab='teams'">{{ txt("报名与队伍","Registration & teams") }}</button><button v-if="detail.can_manage && detail.capabilities.rooms" type="button" @click="emit('create-room',detail.slug)">{{ txt("创建比赛房间","Create match room") }}</button><a href="/practice">{{ txt("前往练习","Practice projects") }} ↗</a></div></header>
        <nav class="detail-tabs" :aria-label="txt('赛事页面','Event pages')"><button v-for="[key,label] in tabs" :key="key" type="button" :aria-pressed="tab===key" @click="tab=key">{{ label }}</button></nav>
        <div v-show="tab==='overview'" class="event-detail-grid">
          <section class="panel"><h2>{{ $t("规则与公告") }}</h2><p class="event-rules">{{ $t(detail.rules) || txt('举办方尚未发布规则。','Rules have not been published yet.') }}</p></section>
<section v-if="detail.capabilities.rooms" class="panel"><h2>{{ $t("参赛信息") }}</h2><p>{{ $t("报名、邀请与队伍管理请在报名页操作，以举办方公布的报名状态为准。") }}</p><p class="muted">{{ txt("按报名表队内序号落座，1 号位为队长。具体局数、选禁及计时以各房间的固定规则为准。","Use your registered seat number; seat 1 is the captain. Game count, draft and timers follow each room’s frozen rules.") }}</p><a href="https://live.2048tables.online/" target="_blank" rel="noopener noreferrer">{{ $t("前往直播大厅 →") }}</a></section>
          <section v-else class="panel"><h2>{{ $t("参赛方式") }}</h2><p>{{ $t("使用已登记的 Table 账号，在对局站进行 3×3 对局。无需进入比赛房间。") }}</p><p class="muted">{{ $t("仅统计比赛期间开始并完成的有效对局，外站导入局不计入。") }}</p><a href="https://play.2048tables.online/" target="_blank" rel="noopener noreferrer">{{ $t("前往对局站 →") }}</a></section>
        </div>
        <details v-show="tab==='manage'" v-if="detail.can_assign_organizer" class="event-admin"><summary>{{ $t("指定赛事举办方") }}</summary><p>{{ $t("当前举办方：") }}{{ $t(detail.organizer ? `Table 用户 ID ${detail.organizer.user_id}` : '尚未指定') }}</p><form @submit.prevent="assignOrganizer(false)"><label>{{ $t("举办方 Table 用户 ID") }}<input v-model="organizerId" type="number" min="1" required :disabled="busy" @input="organizerPreview=null" /></label><button type="submit" :disabled="busy">{{ $t("核对用户") }}</button></form><div v-if="organizerPreview"><p>{{ organizerPreview.display_name }} · ID {{ $t(organizerPreview.user_id) }}</p><p>{{ $t("该用户将能够编辑本赛事信息、导入分组名单及管理报名。") }}</p><button type="button" :disabled="busy" @click="assignOrganizer(true)">{{ $t("确认指定为举办方") }}</button></div></details>
        <EventEnrollment v-show="tab==='teams'" :slug="detail.slug" @changed="enrollmentRevision=$event" />
        <EventStatistics v-show="tab==='results'" v-if="detail.capabilities.statistics" :slug="detail.slug" :roster-revision="enrollmentRevision" />
        <EventFixtures v-if="detail.capabilities.rooms" v-show="tab==='results'||tab==='manage'" :slug="detail.slug" :active="tab==='results'||tab==='manage'" :manage-mode="tab==='manage'" :roster-revision="enrollmentRevision" @manage="tab='manage'" @navigate="emit('navigate',$event)" @changed="emit('refresh')" />
        <details v-show="tab==='results'" v-if="detail.record_candidates?.length" class="event-admin"><summary>{{ $t("赛事纪录候选成绩") }}</summary><p>{{ $t("仅列入本轮获胜队伍的有效完赛成绩，按项目及规则版本分别评选。") }}</p><p v-for="record in detail.record_candidates" :key="`${record.public_key}-${record.game_key}`">{{ record.name }}{{ $t(" · 选手 ID ") }}{{ $t(record.player_user_id) }} · {{ $t(record.project_ref.includes('-race-') ? `用时 ${(record.elapsed_ms/1000).toFixed(2)} 秒` : record.project_ref.includes('cargo') ? `送出 ${record.score} 块` : `成绩 ${record.score}`) }}</p></details>
        <section v-show="tab==='results'" v-if="detail.capabilities.rooms && unboundRooms.length" class="panel event-schedule"><div class="section-heading"><h2>{{ txt("其他比赛房间","Other match rooms") }}</h2><span>{{ unboundRooms.length }}{{ $t(" 场 · 北京时间") }}</span></div>
          <div v-for="(item, index) in unboundRooms" :key="index" class="event-room"><time>{{ $t(scheduleTime(item.schedule?.starts_at) || '时间待定') }}</time><strong>{{ roomMatchTitle(item) }}<small v-if="item.rules"> {{ item.rules.game_count }} {{ txt("局","games") }}</small></strong><b v-if="roomMatchScore(item)">{{ $t(roomMatchScore(item)) }}</b><span>{{ $t(item.schedule?.exception === 'both_late' && item.status !== 'CANCELLED' ? '双方未就位 · 0:0' : statusText?.[item.status] || item.status) }}</span><a v-if="item.room_code" :href="`/rooms/${item.room_code}`" @click.prevent="emit('navigate', `/rooms/${item.room_code}`)">{{ $t("进入房间 →") }}</a></div>
        </section>
        <details v-show="tab==='manage'" v-if="detail.can_manage && detail.capabilities.rooms" class="event-admin"><summary>{{ $t("赛事管理 · 房间关联") }}</summary><p class="muted">{{ $t("已有房间不会自动归入赛事。关联不会改变原房间地址、项目规则或比赛进度。") }}</p><form class="event-link-form" @submit.prevent="linkRoom"><label>{{ $t("已有房间码") }}<input v-model="roomCode" required maxlength="12" /></label><button class="primary-button" :disabled="busy">{{ $t("关联房间") }}</button><button type="button" @click="emit('create-room', detail.slug)">{{ $t("为本赛事创建房间") }}</button></form></details>
        <details v-show="tab==='manage'" v-if="detail.can_manage" class="event-admin"><summary>{{ $t("编辑赛事信息") }}</summary><form @submit.prevent="save"><label>{{ $t("名称") }}<input v-model="settings.name" required minlength="2" maxlength="100" /></label><label>{{ $t("简介") }}<textarea v-model="settings.description" maxlength="1000" /></label><label>{{ $t("规则与公告") }}<textarea v-model="settings.rules" maxlength="10000" rows="5" /></label><label>{{ $t("赛事状态") }}<select v-model="settings.status" :disabled="detail.capabilities.statistics"><option v-for="(label,key) in labels" :key="key" :value="key">{{ $t(label) }}</option></select></label><p class="muted">{{ $t(detail.capabilities.statistics ? '统计型赛事状态按时间窗自动切换；编辑公告不会改变统计时间或成绩算法。' : '赛事状态用于目录展示，不会启动、停止或修改已有对局。修改公告不会改变房间内已固定的项目规则。') }}</p><button class="primary-button" :disabled="busy">{{ $t("保存赛事信息") }}</button></form></details>
      </template>
    </template>
  </section>
</template>

<style scoped>
.event-center{margin-bottom:40px}.event-intro{display:flex;align-items:center;justify-content:space-between;gap:20px}.event-center h1{font-size:clamp(28px,4vw,42px);margin:12px 0}.event-center a{color:inherit}.event-filters{display:flex;gap:8px;margin:24px 0;flex-wrap:wrap}.event-filters button{border:1px solid #ded4c5;border-radius:5px;padding:8px 16px;background:transparent;color:inherit;cursor:pointer}.event-filters button[aria-pressed=true]{background:#a18145;color:#fff;border-color:#a18145}.event-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:20px}.event-card{max-width:560px;text-decoration:none;border:1px solid #e3d8c9;border-radius:8px;overflow:hidden;background:#fffdf8}.event-cover{min-height:160px;background:#242c36;color:#fff4de;padding:24px;display:flex;flex-direction:column;justify-content:center;border-left:5px solid #b39350;gap:12px}.event-cover>span{color:#c4a56a;letter-spacing:3px;font-size:13px}.event-cover strong{font-size:28px}.event-cover small{color:#c3c6cb}.event-card-body{padding:20px}.event-card-body p{line-height:1.7;color:#776e65}.event-card footer{display:flex;justify-content:space-between;gap:12px;font-size:13px}.event-status{font-size:13px;color:#9b7837}.event-detail-title{padding:24px 0}.event-detail-grid{display:grid;grid-template-columns:2fr 1fr;gap:20px}.event-center .panel{margin-bottom:20px}.event-rules{white-space:pre-wrap;line-height:1.9}.event-admin{margin-top:24px;padding:20px;border:1px solid #e3d8c9;border-radius:8px}.event-admin summary{cursor:pointer}.event-admin form{display:grid;gap:14px;margin-top:20px;max-width:720px}.event-admin label{display:grid;gap:8px}.event-admin input,.event-admin textarea{width:100%;box-sizing:border-box;padding:10px;border:1px solid #cdd2d8;border-radius:5px;background:#f5f6f8;color:#40372d;font:inherit}.event-room{display:flex;gap:16px;align-items:center;padding:16px 0;border-bottom:1px solid #e3d8c9}.event-room strong{flex:1}.event-room span{font-size:13px}.event-center a:focus-visible,.event-center button:focus-visible{outline:2px solid #a18145;outline-offset:4px}@media(max-width:650px){.event-detail-grid{grid-template-columns:1fr}.event-room{flex-wrap:wrap}.event-room strong{flex-basis:100%}.event-intro{align-items:flex-start}.event-cover strong{font-size:24px}}
.event-card{background:var(--competition-card);border-color:var(--competition-border)}
.event-card-body p{color:var(--competition-muted)}
.event-center .muted{color:var(--competition-muted)}
.event-status{color:var(--competition-accent)}
.event-admin,.event-room,.event-filters button{border-color:var(--competition-border)}
.event-admin input,.event-admin textarea,.event-admin select{background:var(--competition-page);color:var(--competition-text);border:1px solid var(--competition-border);border-radius:5px;padding:10px;font:inherit}
.event-admin button{padding:10px 14px;border:1px solid var(--competition-border);border-radius:5px;background:var(--competition-card);color:var(--competition-text)}
.event-admin .primary-button{background:var(--competition-accent);color:var(--competition-page)}
</style>

<style scoped>
.event-center .panel{padding:20px;box-sizing:border-box}
.event-center h2{margin:0 0 12px;font-size:22px}
.event-center .muted{color:#817567;line-height:1.7}
.event-center .section-heading{display:flex;justify-content:space-between;align-items:center;gap:12px}
@media(max-width:650px){.event-center .panel{padding:16px}.event-center h2{font-size:20px}}
</style>


<style scoped>
.event-center{--event-gold:var(--competition-accent);max-width:1240px;margin:0 auto 44px}.event-intro{position:relative;overflow:hidden;padding:40px 36px;border:1px solid var(--competition-border);border-radius:18px;background:var(--competition-card);align-items:flex-end;isolation:isolate}.event-intro::after{content:"2048";position:absolute;right:20px;top:8px;font-size:160px;font-weight:900;letter-spacing:-12px;color:var(--event-gold);opacity:.07;z-index:-1;pointer-events:none}.event-intro>div{max-width:700px}.event-intro h1{font-size:clamp(28px,3.5vw,42px);letter-spacing:-.04em;line-height:1.25}.event-intro>a{flex:none;border:1px solid var(--competition-border);padding:12px 16px;border-radius:8px;text-decoration:none;font-size:14px}.eyebrow{color:var(--event-gold);font-size:12px;font-weight:700;letter-spacing:.2em}.lobby-stats{display:grid;grid-template-columns:repeat(3,1fr);margin:22px 0 28px;padding:4px 0}.lobby-stats>div{display:flex;align-items:baseline;gap:12px;padding:0 24px;border-right:1px solid var(--competition-border)}.lobby-stats>div:first-child{padding-left:0}.lobby-stats>div:last-child{border:0}.lobby-stats strong{font-size:28px;font-variant-numeric:tabular-nums}.lobby-stats span{color:var(--competition-muted);font-size:13px}.event-filters{border-bottom:1px solid var(--competition-border);padding-bottom:16px;margin:0 0 22px}.event-filters button{border-radius:24px;font-size:14px}.event-cards{grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:24px}.event-card{max-width:none;border-radius:14px;transition:transform .18s,border-color .18s}.event-card:hover{transform:translateY(-3px);border-color:var(--event-gold)}.event-cover{position:relative;isolation:isolate;overflow:hidden;min-height:180px;padding:28px;border:0;background:linear-gradient(120deg,#283c48,#17252f);box-sizing:border-box}.event-cover::after{content:"";position:absolute;right:-24px;bottom:-45px;width:155px;height:155px;transform:rotate(28deg);border:22px solid #d3b46a1c;border-radius:18px;z-index:-1}.event-card:nth-child(even) .event-cover{background:linear-gradient(120deg,#504333,#302c28)}.event-cover strong{font-size:clamp(23px,2.5vw,30px);line-height:1.3;overflow-wrap:anywhere}.event-cover>span{font-size:11px;letter-spacing:.18em}.event-cover small{font-size:12px;color:#d2d7db}.event-card-body{padding:24px}.event-card-body>p{min-height:3.4em;font-size:14px;margin:16px 0 22px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}.event-card footer{border-top:1px solid var(--competition-border);padding-top:16px;align-items:center}.event-card footer>span:last-child{color:var(--event-gold);font-weight:700}.event-status{display:inline-flex;align-items:center;gap:7px;font-weight:600;font-size:12px}.event-status i{width:6px;height:6px;border-radius:50%;background:currentColor}.event-card.active .event-status{color:light-dark(#357759,#83c6a4)}.event-card.finished .event-status{color:var(--competition-muted)}.event-detail-title{margin:20px 0 0;padding:32px;border:1px solid var(--competition-border);border-radius:16px;background:var(--competition-card)}.event-detail-title h1{font-size:clamp(28px,4vw,40px)}.detail-actions{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-top:24px}.detail-actions button{padding:11px 18px;border:1px solid var(--competition-border);border-radius:7px;font:inherit;color:inherit;background:transparent;cursor:pointer}.detail-actions .primary-button{background:var(--event-gold);color:var(--competition-page);border-color:var(--event-gold)}.detail-actions a{font-size:14px;padding:10px}.detail-tabs{display:flex;gap:8px;overflow-x:auto;border-bottom:1px solid var(--competition-border);margin:20px 0 28px}.detail-tabs button{flex:none;font:inherit;font-size:14px;padding:14px 18px;border:0;border-bottom:3px solid transparent;background:none;color:var(--competition-muted);cursor:pointer}.detail-tabs button[aria-pressed=true]{border-color:var(--event-gold);color:var(--competition-text);font-weight:700}.event-detail-grid{grid-template-columns:minmax(0,1.8fr) minmax(0,1fr)}.event-center .panel{padding:24px;border:1px solid var(--competition-border);border-radius:12px;background:var(--competition-card)}.event-center .muted{color:var(--competition-muted)}.event-center h2{font-size:20px}.event-rules{font-size:14px;overflow-wrap:anywhere;line-height:1.9}.event-admin{background:var(--competition-card);border-radius:12px}.event-room{display:grid;grid-template-columns:minmax(105px,.8fr) minmax(150px,2fr) auto auto auto;gap:18px;font-size:14px}.event-room time,.event-room small{color:var(--competition-muted);font-size:12px}.event-room>a{color:var(--event-gold);white-space:nowrap}.event-room:last-child{border-bottom:0}.event-center .empty-state{padding:40px;text-align:center;border:1px dashed var(--competition-border);border-radius:12px;color:var(--competition-muted)}
@media(max-width:650px){.event-intro{padding:24px;display:block}.event-intro>a{display:inline-block;margin-top:12px}.event-intro::after{font-size:100px;top:auto;bottom:0}.lobby-stats{gap:10px}.lobby-stats>div{padding:0 10px;display:grid;gap:5px}.lobby-stats strong{font-size:24px}.lobby-stats span{font-size:12px}.event-detail-title{padding:22px}.detail-tabs button{padding:12px 14px}.event-detail-grid{grid-template-columns:1fr}.event-room{grid-template-columns:1fr auto;gap:9px}.event-room strong{grid-column:1/-1;grid-row:1}.event-cover{min-height:155px}.event-card-body{padding:20px}.event-center .panel{padding:18px}.event-center .section-heading{flex-wrap:wrap}.event-card:hover{transform:none}}
</style>
