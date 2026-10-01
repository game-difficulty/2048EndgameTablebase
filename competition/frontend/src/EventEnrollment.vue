<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { api } from './api.js';
import { t } from './i18n.js';
import { userFacingError } from './errorMessages.js';

const props = defineProps({ slug: { type: String, required: true } });
const emit = defineEmits(['changed']);
const state = ref(null), error = ref(''), notice = ref(''), busy = ref(false);
const teamName = ref(''), inviteUser = ref(''), unlockReason = ref('');
const mode = ref('solo'), capacity = ref(0), open = ref(false);
const csv = ref(''), preview = ref(null);
const identityMode = ref('username');
let disposed = false;
let pollTimer;
let lastEmittedRevision;
const labels = { solo: '单人报名', self_team: '自由组队报名', organizer_team: '个人报名 · 举办方分队' };
const mine = computed(() => state.value?.entries.find(e => e.user_id === state.value.me.user_id));
const myTeam = computed(() => state.value?.teams.find(t => t.id === mine.value?.team_id));
const captain = computed(() => myTeam.value?.captain_user_id === state.value?.me.user_id && !!myTeam.value);
const editable = computed(() => state.value?.registration_open && !state.value?.registration_locked && !state.value?.roster_locked);
const members = team => state.value.entries.filter(e => e.team_id === team.id);
function update(value, preserveDraft = false) {
  if (disposed) return;
  if (state.value && value.revision < state.value.revision) return;
  if (preview.value && preview.value.revision !== value.revision) preview.value = null;
  state.value = value;
  if (!preserveDraft) { mode.value = value.mode; capacity.value = value.capacity; open.value = !!value.registration_open; }
  if (value.revision !== lastEmittedRevision) { lastEmittedRevision = value.revision; emit('changed', value.revision); }
}
async function load(preserveDraft = false) { try { update(await api.enrollment(props.slug), preserveDraft); } catch(cause) { error.value = userFacingError(cause); } }
async function act(action, values = {}) {
  if (busy.value) return;
  if (['withdraw','disband_team','lock_registration','lock_roster'].includes(action) && !window.confirm(t({withdraw:'确定退出报名？',disband_team:'解散后所有队员仍保留报名，但会变为未组队。确定继续？',lock_registration:'锁定参赛人员后不能自主报名或退出；举办方仍可调整分组。确定？',lock_roster:'锁定最终名单后，报名、组队和导入均停止。确定名单已经核对完毕？'}[action]))) return;
  busy.value = true; error.value = ''; notice.value = '';
  try { update(await api.enrollmentAction(props.slug, { action, revision: state.value.revision, ...values })); preview.value = null; notice.value = '操作已保存。'; }
  catch(cause) { error.value = userFacingError(cause); if (cause.code === 'ROSTER_CHANGED') await load(); }
  finally { busy.value = false; }
}
async function importList(save = false) {
  if (busy.value) return;
  busy.value = true; error.value = ''; notice.value = '';
  try {
    let entries = preview.value?.entries;
    if (!save) preview.value = null;
    if (!save) entries = csv.value.trim().split(/\r?\n/).filter(line => line.trim()).map((line,index) => {
      const [id, team = '', ext = '0', cap = '0', pos = '', extra] = line.split(/[,，\t]/).map(part => part.trim());
      if (!id || (identityMode.value === 'user_id' && (!/^[1-9]\d*$/.test(id) || !Number.isSafeInteger(Number(id)))) || extra !== undefined || !['0','1'].includes(ext) || !['0','1'].includes(cap) || (pos && !['1','2','3'].includes(pos))) throw new Error(`第 ${index+1} 行有误：${t(identityMode.value === 'username' ? '当前用户名' : '用户 ID')},${t('队名（可空）,外援0或1,队长0或1,队内序号（团队对战填1/2/3）。')}`);
      return { [identityMode.value]:identityMode.value === 'username' ? id : Number(id), team_name:team, is_external:ext==='1', captain:cap==='1', position:pos ? Number(pos) : null };
    });
    if (!entries?.length || entries.length > 500) throw new Error('请提供 1–500 位选手。');
    const result = await api.importEnrollment(props.slug, { entries, revision: save ? preview.value.revision : state.value.revision, dry_run:!save });
    if (save) { preview.value = null; await load(); notice.value = '名单已保存。未分组的选手仍为待分组状态。'; }
    else preview.value = result;
  } catch(cause) { error.value = userFacingError(cause); if(cause.code==='ROSTER_CHANGED') await load(); }
  finally { busy.value = false; }
}
function exportToEditor() {
  identityMode.value = 'user_id';
  csv.value = state.value.entries.map(e => {
    const team = state.value.teams.find(t => t.id === e.team_id);
    return [e.user_id,team?.name || '',e.is_external?1:0,team?.captain_user_id===e.user_id?1:0,e.position || ''].join(',');
  }).join('\n'); preview.value = null;
}
onMounted(() => { load(); pollTimer = setInterval(() => { if (!document.hidden && !busy.value) load(true); }, 15000); });
onBeforeUnmount(() => { disposed = true; clearInterval(pollTimer); });
</script>

<template>
  <section class="enrollment-panel">
    <header><h2>{{ $t("报名与参赛名单") }}</h2><button type="button" :disabled="busy" @click="load()">{{ $t("刷新") }}</button></header>
    <p v-if="error" class="alert" role="alert">{{ $t(error) }}</p><p v-if="notice" role="status">{{ $t(notice) }}</p>
    <template v-if="state">
      <p class="enrollment-meta">{{ $t(labels[state.mode]) }} · {{ $t(state.entries.length) }}{{ $t(state.capacity ? `/${state.capacity}` : '') }}{{ $t(" 人 · ") }}{{ $t(state.roster_locked ? '最终名单已锁定' : state.registration_locked ? '参赛人员已锁定，分组可调整' : state.registration_open ? '报名开放' : '报名未开放') }}</p>
      <p v-if="!state.me.user_id"><a href="https://2048tables.online/">{{ $t("登录 Table 账号后报名或处理邀请 →") }}</a></p>
      <template v-else>
        <p class="enrollment-meta">{{ $t("你的 Table 用户 ID：") }}{{ $t(state.me.user_id) }}。<template v-if="state.mode==='self_team'">{{ $t("可将此 ID 提供给队长用于邀请。") }}</template></p>
        <p v-if="mine">{{ $t("你已") }}{{ $t(mine.source === 'imported' ? '由举办方登记' : '报名') }}：{{ myTeam?.name || (state.mode === 'solo' ? '单人参赛' : '待分组 / 待组队') }}。</p>
        <div class="enrollment-actions" v-if="editable"><button v-if="!mine" :disabled="busy" @click="act('signup')">{{ $t("报名参赛") }}</button><button v-else-if="!myTeam || state.mode !== 'self_team'" :disabled="busy" @click="act('withdraw')">{{ $t("退出报名") }}</button></div>
        <template v-if="state.mode === 'self_team'">
          <form v-if="!myTeam && editable" @submit.prevent="act('create_team',{name:teamName})"><label>{{ $t("队伍名称") }}<input v-model="teamName" required maxlength="40" /></label><button :disabled="busy">{{ $t("创建队伍并报名") }}</button></form>
          <div v-for="invite in state.invitations.filter(i=>i.user_id===state.me.user_id)" :key="invite.team_id" class="invite-row">「{{ invite.team_name }}{{ $t("」邀请你加入 ") }}<button :disabled="busy || !editable" @click="act('accept_invite',{team_id:invite.team_id})">{{ $t("接受") }}</button><button :disabled="busy || !editable" @click="act('decline_invite',{team_id:invite.team_id})">{{ $t("拒绝") }}</button></div>
          <section v-if="myTeam" class="my-team"><h3>{{ $t("我的队伍 · ") }}{{ myTeam.name }} · {{ $t(myTeam.submitted ? '已提交报名' : '待队长提交') }}</h3>
            <p>{{ members(myTeam).map(e=>e.display_name).join('、') }}（{{ $t(members(myTeam).length) }}/{{ $t(state.team_size) }}）</p>
            <template v-if="captain"><form v-if="!myTeam.submitted && editable" @submit.prevent="act('invite',{team_id:myTeam.id,user_id:Number(inviteUser)})"><label>{{ $t("队员 Table 用户 ID") }}<input v-model="inviteUser" type="number" min="1" required /></label><button :disabled="busy">{{ $t("发出邀请") }}</button></form>
              <p v-for="invite in state.invitations.filter(i=>i.team_id===myTeam.id)" :key="invite.user_id">{{ $t("等待 ") }}{{ invite.display_name }}{{ $t(" 接受 ") }}<button :disabled="busy || !editable" @click="act('cancel_invite',{team_id:myTeam.id,user_id:invite.user_id})">{{ $t("取消邀请") }}</button></p>
              <div class="enrollment-actions"><button :disabled="busy || state.roster_locked || (!myTeam.submitted && members(myTeam).length !== state.team_size)" @click="act(myTeam.submitted?'unsubmit_team':'submit_team',{team_id:myTeam.id})">{{ $t(myTeam.submitted ? '撤回队伍报名' : '全员确认，提交队伍报名') }}</button><button :disabled="busy || !editable" @click="act('disband_team',{team_id:myTeam.id})">{{ $t("解散队伍") }}</button></div>
            </template><button v-else :disabled="busy || !editable" @click="act('leave_team',{team_id:myTeam.id})">{{ $t("离开队伍（保留个人报名）") }}</button>
          </section>
        </template>
      </template>
      <div class="enrolled-teams"><article v-for="team in state.teams" :key="team.id"><h3>{{ team.name }} <small>{{ $t(state.roster_locked ? '已锁定' : state.mode === 'self_team' ? (team.submitted ? '已提交' : '待队长提交') : '分组草案') }}</small></h3><p v-for="member in members(team)" :key="member.user_id">{{ $t(member.position ? `${member.position}号 · ` : '') }}{{ member.display_name }}{{ $t(team.captain_user_id === member.user_id ? ' · 队长' : '') }}{{ $t(member.is_external ? ' · 外援' : '') }}</p></article></div>
      <p v-if="state.entries.some(e=>!e.team_id)">{{ $t(state.mode==='solo'?'参赛选手':'待分组选手') }}：{{ state.entries.filter(e=>!e.team_id).map(e=>e.display_name+(e.is_external?'（外援）':'')).join('、') }}</p>
      <p v-if="!state.entries.length" class="enrollment-meta">{{ $t("暂无报名或导入名单。") }}</p>
      <details v-if="state.me.can_manage" class="enrollment-admin"><summary>{{ $t("举办方 · 报名设置、导入与锁定") }}</summary>
        <form v-if="!state.roster_locked && !state.registration_locked" @submit.prevent="act('settings',{mode,capacity:Number(capacity),registration_open:open})"><label>{{ $t("报名方式") }}<select v-model="mode" :disabled="state.fixed_policy || !!state.entries.length"><option v-for="(label,key) in labels" :key="key" :value="key">{{ $t(label) }}</option></select></label><label>{{ $t("人数上限（0 不限）") }}<input v-model="capacity" type="number" min="0" max="10000" :disabled="state.fixed_policy" /></label><label class="checkbox-label"><input v-model="open" type="checkbox" />{{ $t("开放报名") }}</label><button :disabled="busy">{{ $t("保存设置") }}</button></form>
        <div class="enrollment-actions" v-if="!state.roster_locked"><button v-if="!state.registration_locked" :disabled="busy" @click="act('lock_registration')">{{ $t("锁定参赛人员") }}</button><button :disabled="busy" @click="act('lock_roster')">{{ $t("锁定最终名单") }}</button></div>
        <form v-if="state.registration_locked || state.roster_locked" @submit.prevent="act('unlock',{reason:unlockReason})"><label>{{ $t("解锁原因") }}<input v-model="unlockReason" minlength="4" maxlength="500" required /></label><button :disabled="busy">{{ $t("解锁（不会自动开放报名）") }}</button></form>
        <template v-if="!state.roster_locked"><h3>{{ $t("导入 / 调整名单") }}</h3><p>{{ $t("每行：当前用户名或用户ID（按导入方式选择）,队名（未分组留空）,外援0或1,队长0或1,队内序号（团队对战填1/2/3）。仅填用户名或 ID 也可导入。用户名按主站规则匹配当前有效账号，不匹配历史用户名；任一行匹配失败则整批不导入。保存将整体替换当前名单，并取消旧邀请；锁定参赛人员后只能调整同一批人员的分组。自由组队的分组须各指定一名队长，导入后仍需队长提交。") }}</p><label>{{ $t("导入方式") }}<select v-model="identityMode" :disabled="busy" @change="preview=null"><option value="username">{{ $t("当前用户名") }}</option><option value="user_id">{{ $t("用户 ID") }}</option></select></label><button :disabled="busy" @click="exportToEditor">{{ $t("载入当前名单编辑") }}</button><label>{{ $t("名单") }}<textarea v-model="csv" rows="8" :disabled="busy" :placeholder="identityMode === 'username' ? 'Player One,,0,0\nPlayer Two,,1,0' : '123,,0,0\n456,,1,0'" @input="preview=null" /></label><button :disabled="busy || !csv.trim()" @click="importList(false)">{{ $t("校验并预览") }}</button>
          <div v-if="preview"><h3>{{ $t("预览 · 尚未保存") }}</h3><p v-for="entry in preview.entries" :key="entry.user_id">{{ $t(entry.position ? `${entry.position}号 · ` : '') }}{{ entry.display_name }}（ID {{ $t(entry.user_id) }}） · {{ entry.team_name || '未分组' }}{{ $t(entry.is_external ? ' · 外援' : '') }}{{ $t(entry.captain ? ' · 队长' : '') }}</p><button :disabled="busy" @click="importList(true)">{{ $t("确认替换为这 ") }}{{ $t(preview.entries.length) }}{{ $t(" 位选手") }}</button></div>
        </template>
        <details><summary>{{ $t("最近操作记录") }}</summary><p v-for="item in state.audit" :key="item.revision">#{{ $t(item.revision) }}{{ $t(" · 管理/操作账号 ") }}{{ $t(item.actor_user_id) }} · {{ $t(item.action) }} · {{ $t(item.created_at) }}<span v-if="item.action==='unlock'"> · {{ $t(JSON.parse(item.payload_json).reason) }}</span></p></details>
      </details>
    </template>
  </section>
</template>

<style scoped>
.enrollment-panel{padding:22px;border:1px solid #e3d8c9;border-radius:8px;margin:24px 0;background:#fffdf8}.enrollment-panel header{display:flex;justify-content:space-between;align-items:center;gap:12px}.enrollment-panel h2{margin:0;font-size:22px}.enrollment-panel p{line-height:1.7}.enrollment-meta,small{color:#817567;font-size:13px}.enrollment-panel button{padding:9px 14px;background:#fffdf8;color:inherit;border:1px solid #cbb895;border-radius:5px;cursor:pointer}.enrollment-panel button:disabled{opacity:.5;cursor:default}.enrollment-panel form{display:flex;align-items:end;gap:12px;flex-wrap:wrap;margin:18px 0}.enrollment-panel label{display:grid;gap:8px}.enrollment-panel input,.enrollment-panel select,.enrollment-panel textarea{padding:10px;border:1px solid #cdd2d8;border-radius:5px;font:inherit;max-width:100%;box-sizing:border-box;background:#f5f6f8;color:inherit}.enrollment-actions{display:flex;gap:10px;flex-wrap:wrap;margin:15px 0}.enrollment-admin{margin-top:22px;border-top:1px solid #e3d8c9;padding-top:18px}.enrollment-admin summary{cursor:pointer}.enrollment-admin textarea{width:100%;margin:12px 0}.enrolled-teams{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}.enrolled-teams article,.my-team{padding:16px;border:1px solid #e3d8c9;border-radius:6px;margin:16px 0}.enrolled-teams h3{margin:0}.checkbox-label{display:flex!important;align-items:center;padding:10px}.invite-row{padding:12px 0}.enrollment-panel a{color:inherit}@media(max-width:600px){.enrollment-panel{padding:16px}.enrollment-panel form{align-items:stretch;flex-direction:column}.enrollment-panel input:not([type=checkbox]),.enrollment-panel select{width:100%}.enrolled-teams{grid-template-columns:1fr}}
.enrollment-panel{background:var(--competition-card);border-color:var(--competition-border)}
.enrollment-meta,.enrollment-panel small{color:var(--competition-muted)}
.enrollment-panel button{background:var(--competition-card);border-color:var(--competition-border)}
.enrollment-panel input,.enrollment-panel select,.enrollment-panel textarea{background:var(--competition-page);border-color:var(--competition-border);color:var(--competition-text)}
.enrollment-admin,.enrolled-teams article,.my-team{border-color:var(--competition-border)}
</style>
