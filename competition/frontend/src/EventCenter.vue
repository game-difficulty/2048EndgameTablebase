<script setup>
import { computed, ref, watch, onMounted, onBeforeUnmount } from 'vue';
import { scheduleTime, roomMatchTitle, roomMatchScore } from './scheduleDisplay.js';
import { api } from './api.js';
import { userFacingError } from './errorMessages.js';
import EventStatistics from './EventStatistics.vue';
import EventEnrollment from './EventEnrollment.vue';

const props = defineProps({ slug: { type: String, default: '' }, events: { type: Array, default: () => [] }, canCreate: Boolean, statusText: Object });
const emit = defineEmits(['navigate', 'refresh', 'create-room']);
const detail = ref(null);
const loading = ref(false);
const error = ref('');
const busy = ref(false);
const filter = ref('all');
const roomCode = ref('');
const draft = ref({ slug: '', name: '', description: '', rules: '' });
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
      <div class="event-intro"><div><p class="eyebrow">2048 · COMPETITION</p><h1>{{ $t("赛事中心") }}</h1><p class="muted">{{ $t("选择赛事，查看规则与比赛安排。") }}</p></div><a href="/practice">{{ $t("项目练习 →") }}</a></div>
      <nav class="event-filters" :aria-label='$t("赛事状态")'><button v-for="(label, key) in { all: '全部赛事', ...labels }" :key="key" :aria-pressed="filter === key" @click="filter = key">{{ $t(label) }}</button></nav>
      <div class="event-cards">
        <a v-for="event in shown" :key="event.slug" :href="`/events/${event.slug}`" class="event-card" @click.prevent="emit('navigate', `/events/${event.slug}`)">
          <div class="event-cover"><span>2048</span><strong>{{ event.name }}</strong><small>{{ $t(event.label) }}</small></div>
          <div class="event-card-body"><span class="event-status">{{ $t(labels[event.status]) }}</span><p>{{ $t(event.description || '进入赛事查看详情。') }}</p><footer><span>{{ $t(event.capabilities.statistics ? '名单导入 · 成绩统计' : `${event.room_count} 个比赛房间`) }}</span><span>{{ $t("查看赛事 →") }}</span></footer></div>
        </a>
      </div>
      <p v-if="!shown.length" class="empty-state">{{ $t("暂无此状态的赛事。") }}</p>
      <details v-if="canCreate" class="event-admin"><summary>{{ $t("创建赛事") }}</summary><form @submit.prevent="create">
        <label>{{ $t("赛事名称") }}<input v-model="draft.name" required minlength="2" maxlength="100" /></label>
        <label>{{ $t("赛事地址标识") }}<input v-model="draft.slug" required minlength="2" maxlength="64" pattern="[a-z0-9]+(-[a-z0-9]+)*" :placeholder='$t("例如 summer-cup-1")' /></label>
        <label>{{ $t("简介") }}<textarea v-model="draft.description" maxlength="1000" /></label>
        <label>{{ $t("规则与公告") }}<textarea v-model="draft.rules" maxlength="10000" rows="4" /></label>
        <p class="muted">{{ $t("当前创建三人团队选 Ban 赛事。默认关闭报名，可在赛事管理中设置报名方式并开放。") }}</p>
        <button class="primary-button" :disabled="busy">{{ $t("创建赛事") }}</button>
      </form></details>
    </template>
    <template v-else>
      <a href="/events" @click.prevent="emit('navigate', '/events')">{{ $t("← 全部赛事") }}</a>
      <p v-if="loading" class="empty-state">{{ $t("正在加载赛事…") }}</p>
      <template v-else-if="detail">
        <header class="event-detail-title"><span class="event-status">{{ $t(labels[detail.status]) }} · {{ $t(detail.label) }}</span><h1>{{ detail.name }}</h1><p class="muted">{{ detail.description }}</p></header>
        <div class="event-detail-grid">
          <section class="panel"><h2>{{ $t("规则与公告") }}</h2><p class="event-rules">{{ detail.rules || '举办方尚未发布规则。' }}</p></section>
<section v-if="detail.capabilities.rooms" class="panel"><h2>{{ $t("参赛信息") }}</h2><p>{{ $t("报名、邀请与队伍提交请在下方操作，以举办方公布的报名状态为准。") }}</p><p class="muted">{{ $t("请按报名表队内序号落座。开战后超过 15 分钟，仅一方就位则该方 3:0 获胜；双方均未就位则 0:0。") }}</p><a href="https://live.2048tables.online/" target="_blank" rel="noopener noreferrer">{{ $t("前往直播大厅 →") }}</a></section>
          <section v-else class="panel"><h2>{{ $t("参赛方式") }}</h2><p>{{ $t("使用已登记的 Table 账号，在对局站进行 3×3 对局。无需进入比赛房间。") }}</p><p class="muted">{{ $t("仅统计比赛期间开始并完成的有效对局，外站导入局不计入。") }}</p><a href="https://play.2048tables.online/" target="_blank" rel="noopener noreferrer">{{ $t("前往对局站 →") }}</a></section>
        </div>
        <details v-if="detail.can_assign_organizer" class="event-admin"><summary>{{ $t("指定赛事举办方") }}</summary><p>{{ $t("当前举办方：") }}{{ $t(detail.organizer ? `Table 用户 ID ${detail.organizer.user_id}` : '尚未指定') }}</p><form @submit.prevent="assignOrganizer(false)"><label>{{ $t("举办方 Table 用户 ID") }}<input v-model="organizerId" type="number" min="1" required :disabled="busy" @input="organizerPreview=null" /></label><button type="submit" :disabled="busy">{{ $t("核对用户") }}</button></form><div v-if="organizerPreview"><p>{{ organizerPreview.display_name }} · ID {{ $t(organizerPreview.user_id) }}</p><p>{{ $t("该用户将能够编辑本赛事信息、导入分组名单及管理报名。") }}</p><button type="button" :disabled="busy" @click="assignOrganizer(true)">{{ $t("确认指定为举办方") }}</button></div></details>
        <EventEnrollment :slug="detail.slug" @changed="enrollmentRevision=$event" />
        <EventStatistics v-if="detail.capabilities.statistics" :slug="detail.slug" :roster-revision="enrollmentRevision" />
        <details v-if="detail.record_candidates?.length" class="event-admin"><summary>{{ $t("赛事纪录候选成绩") }}</summary><p>{{ $t("仅列入本轮获胜队伍的有效完赛成绩，按项目及规则版本分别评选。") }}</p><p v-for="record in detail.record_candidates" :key="`${record.public_key}-${record.game_key}`">{{ record.name }}{{ $t(" · 选手 ID ") }}{{ $t(record.player_user_id) }} · {{ $t(record.project_ref.includes('-race-') ? `用时 ${(record.elapsed_ms/1000).toFixed(2)} 秒` : record.project_ref.includes('cargo') ? `送出 ${record.score} 块` : `成绩 ${record.score}`) }}</p></details>
        <section v-if="detail.capabilities.rooms" class="panel"><div class="section-heading"><h2>{{ $t("赛程与结果") }}</h2><span>{{ $t(detail.room_count) }}{{ $t(" 场 · 北京时间") }}</span></div>
          <div v-for="(item, index) in detail.rooms" :key="index" class="event-room"><time>{{ $t(scheduleTime(item.schedule?.starts_at) || '时间待定') }}</time><strong>{{ roomMatchTitle(item) }}</strong><b v-if="roomMatchScore(item)">{{ $t(roomMatchScore(item)) }}</b><span>{{ $t(item.schedule?.exception === 'both_late' && item.status !== 'CANCELLED' ? '双方未就位 · 0:0' : statusText?.[item.status] || item.status) }}</span><a v-if="item.room_code" :href="`/rooms/${item.room_code}`" @click.prevent="emit('navigate', `/rooms/${item.room_code}`)">{{ $t("进入房间 →") }}</a></div>
          <p v-if="!detail.rooms.length" class="muted">{{ $t("举办方尚未关联比赛房间。") }}</p>
        </section>
        <details v-if="detail.can_manage && detail.capabilities.rooms" class="event-admin"><summary>{{ $t("赛事管理 · 房间关联") }}</summary><p class="muted">{{ $t("已有房间不会自动归入赛事。关联不会改变原房间地址、项目规则或比赛进度。") }}</p><form class="event-link-form" @submit.prevent="linkRoom"><label>{{ $t("已有房间码") }}<input v-model="roomCode" required maxlength="12" /></label><button class="primary-button" :disabled="busy">{{ $t("关联房间") }}</button><button type="button" @click="emit('create-room', detail.slug)">{{ $t("为本赛事创建房间") }}</button></form></details>
        <details v-if="detail.can_manage" class="event-admin"><summary>{{ $t("编辑赛事信息") }}</summary><form @submit.prevent="save"><label>{{ $t("名称") }}<input v-model="settings.name" required minlength="2" maxlength="100" /></label><label>{{ $t("简介") }}<textarea v-model="settings.description" maxlength="1000" /></label><label>{{ $t("规则与公告") }}<textarea v-model="settings.rules" maxlength="10000" rows="5" /></label><label>{{ $t("赛事状态") }}<select v-model="settings.status" :disabled="detail.capabilities.statistics"><option v-for="(label,key) in labels" :key="key" :value="key">{{ $t(label) }}</option></select></label><p class="muted">{{ $t(detail.capabilities.statistics ? '统计型赛事状态按时间窗自动切换；编辑公告不会改变统计时间或成绩算法。' : '赛事状态用于目录展示，不会启动、停止或修改已有对局。修改公告不会改变房间内已固定的项目规则。') }}</p><button class="primary-button" :disabled="busy">{{ $t("保存赛事信息") }}</button></form></details>
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
