<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { api } from './api.js';
import { t } from './i18n.js';
import { userFacingError } from './errorMessages.js';
import { parseRosterImport, selectRosterCaptain, rosterImportError, rosterCsvCell } from './rosterImport.js';

const props = defineProps({ slug: { type: String, required: true } });
const emit = defineEmits(['changed']);
const state = ref(null), error = ref(''), notice = ref(''), busy = ref(false);
const teamName = ref(''), inviteUser = ref(''), unlockReason = ref('');
const mode = ref('solo'), capacity = ref(0), open = ref(false);
const csv = ref(''), preview = ref(null);
const identityMode = ref('username');
const importFormat = ref('teams');
const managementTab = ref('import');
const managementPanel = ref(null);
const rosterSearch = ref('');
const previewTeams = computed(() => {
  const groups = new Map();
  for (const entry of preview.value?.entries || []) {
    if (!groups.has(entry.team_name)) groups.set(entry.team_name, []);
    groups.get(entry.team_name).push(entry);
  }
  return Array.from(groups, ([name, entries]) => ({ name, entries: entries.slice().sort((a, b) => (a.position || 0) - (b.position || 0)) }));
});
const importPlaceholder = computed(() => importFormat.value === 'teams'
  ? ['Team A', ...Array.from({ length: state.value?.team_size || 3 }, (_, i) => identityMode.value === 'username' ? `Player ${i + 1}` : `${i + 1}`)].join('\t')
  : identityMode.value === 'username' ? 'Player One,,0,0\nPlayer Two,,1,0' : '123,,0,0\n456,,1,0');
let disposed = false;
let pollTimer;
let lastEmittedRevision;
const labels = { solo: '单人报名', self_team: '自由组队报名', organizer_team: '个人报名 · 举办方分队' };
const mine = computed(() => state.value?.entries.find(e => e.user_id === state.value.me.user_id));
const myTeam = computed(() => state.value?.teams.find(t => t.id === mine.value?.team_id));
const captain = computed(() => myTeam.value?.captain_user_id === state.value?.me.user_id && !!myTeam.value);
const editable = computed(() => state.value?.registration_open && !state.value?.registration_locked && !state.value?.roster_locked);
const statusLabel = computed(() => state.value?.roster_locked ? '最终名单已锁定' : state.value?.registration_locked ? '人员已锁定' : state.value?.registration_open ? '报名开放' : '报名未开放');
const unassigned = computed(() => state.value?.entries.filter(entry => !entry.team_id) || []);
const invitations = computed(() => state.value?.invitations.filter(invite => invite.user_id === state.value.me.user_id) || []);
const pendingInvites = computed(() => state.value?.invitations.filter(invite => invite.team_id === myTeam.value?.id) || []);
const members = team => state.value.entries.filter(e => e.team_id === team.id).sort((a,b) => (a.position || 99) - (b.position || 99));
const matchesSearch = value => String(value).toLocaleLowerCase().includes(rosterSearch.value.trim().toLocaleLowerCase());
const shownTeams = computed(() => state.value?.teams.filter(team => matchesSearch(team.name) || members(team).some(entry => matchesSearch(entry.display_name) || matchesSearch(entry.user_id))) || []);
const shownUnassigned = computed(() => unassigned.value.filter(entry => matchesSearch(entry.display_name) || matchesSearch(entry.user_id)));
function update(value, preserveDraft = false) {
  if (disposed) return;
  if (state.value && value.revision < state.value.revision) return;
  if (value.mode === 'solo') importFormat.value = 'advanced';
  if (value.roster_locked && managementTab.value === 'import') managementTab.value = 'locks';
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
  let sourceRows = [];
  try {
    let entries = preview.value?.entries;
    if (!save) preview.value = null;
    if (!save) ({ entries, sourceRows } = parseRosterImport(csv.value, { format: importFormat.value, identityMode: identityMode.value, teamSize: state.value.team_size }));
    if (!entries?.length || entries.length > 500) throw new Error('请提供 1–500 位选手。');
    const result = await api.importEnrollment(props.slug, { entries, revision: save ? preview.value.revision : state.value.revision, dry_run:!save });
    if (save) { preview.value = null; await load(); notice.value = '名单已保存。未分组的选手仍为待分组状态。'; }
    else preview.value = result;
  } catch(cause) { error.value = rosterImportError(userFacingError(cause), sourceRows); if(cause.code==='ROSTER_CHANGED') await load(); }
  finally { busy.value = false; }
}
function exportToEditor() {
  importFormat.value = 'advanced';
  identityMode.value = 'user_id';
  csv.value = state.value.entries.map(e => {
    const team = state.value.teams.find(t => t.id === e.team_id);
    return [e.user_id,team?.name || '',e.is_external?1:0,team?.captain_user_id===e.user_id?1:0,e.position || ''].map(rosterCsvCell).join(',');
  }).join('\n'); preview.value = null;
}
function openManagement() {
  if (!managementPanel.value) return;
  managementPanel.value.open = true;
  managementPanel.value.scrollIntoView({ behavior:'smooth', block:'start' });
}
onMounted(() => { load(); pollTimer = setInterval(() => { if (!document.hidden && !busy.value) load(true); }, 15000); });
onBeforeUnmount(() => { disposed = true; clearInterval(pollTimer); });
</script>

<template>
  <section class="enrollment-panel">
    <header class="enrollment-heading">
      <div><p class="section-kicker">{{ $t('参赛阵容') }}</p><h2>{{ $t("报名与参赛名单") }}</h2></div>
      <div class="heading-actions"><span v-if="state" class="status-badge" :class="{ active:editable }">{{ $t(statusLabel) }}</span><button v-if="state?.me.can_manage" type="button" @click="openManagement">{{ $t('管理名单') }}</button><button type="button" :disabled="busy" @click="load()">{{ $t("刷新") }}</button></div>
    </header>
    <p v-if="error" class="enrollment-feedback error" role="alert">{{ $t(error) }}</p><p v-if="notice" class="enrollment-feedback success" role="status">{{ $t(notice) }}</p>
    <template v-if="state">
      <div class="enrollment-overview">
        <div><strong>{{ state.entries.length }}<small v-if="state.capacity"> / {{ state.capacity }}</small></strong><span>{{ $t('参赛选手') }}</span></div>
        <div v-if="state.mode!=='solo'"><strong>{{ state.teams.length }}</strong><span>{{ $t('参赛队伍') }}</span></div>
        <div v-if="state.mode!=='solo'"><strong>{{ unassigned.length }}</strong><span>{{ $t('待分组选手') }}</span></div>
        <p>{{ $t(labels[state.mode]) }}<span v-if="state.mode!=='solo'">{{ $t(`每队 ${state.team_size} 人。`) }}</span></p>
      </div>
      <details class="personal-registration" :open="invitations.length>0">
        <summary><span><strong>{{ $t('我的报名') }}</strong><small>{{ !state.me.user_id ? $t('登录后参与报名') : myTeam?.name || $t(mine ? (state.mode==='solo'?'已报名':'待分组 / 待组队') : '尚未报名') }}</small></span><span class="summary-hint">{{ $t('查看与操作') }}</span></summary>
        <div class="personal-content">
          <a v-if="!state.me.user_id" href="https://2048tables.online/">{{ $t("登录 Table 账号后报名或处理邀请 →") }}</a>
          <template v-else>
            <p class="enrollment-meta">Table ID {{ state.me.user_id }}<span v-if="state.mode==='self_team'"> · {{ $t("可将此 ID 提供给队长用于邀请。") }}</span></p>
            <div class="enrollment-actions" v-if="editable"><button v-if="!mine" class="enrollment-primary" :disabled="busy" @click="act('signup')">{{ $t("报名参赛") }}</button><button v-else-if="!myTeam || state.mode !== 'self_team'" :disabled="busy" @click="act('withdraw')">{{ $t("退出报名") }}</button></div>
            <template v-if="state.mode === 'self_team'">
              <form v-if="!myTeam && editable" @submit.prevent="act('create_team',{name:teamName})"><label>{{ $t("队伍名称") }}<input v-model="teamName" required maxlength="40" /></label><button :disabled="busy">{{ $t("创建队伍并报名") }}</button></form>
              <div v-for="invite in invitations" :key="invite.team_id" class="invite-row"><span><strong>{{ invite.team_name }}</strong> · {{ $t('邀请你加入队伍') }}</span><div class="enrollment-actions"><button class="enrollment-primary" :disabled="busy || !editable" @click="act('accept_invite',{team_id:invite.team_id})">{{ $t("接受") }}</button><button :disabled="busy || !editable" @click="act('decline_invite',{team_id:invite.team_id})">{{ $t("拒绝") }}</button></div></div>
              <section v-if="myTeam">
                <p>{{ members(myTeam).map(entry=>entry.display_name).join('、') }} <small>{{ members(myTeam).length }}/{{ state.team_size }}</small></p>
                <template v-if="captain">
                  <form v-if="!myTeam.submitted && editable" @submit.prevent="act('invite',{team_id:myTeam.id,user_id:Number(inviteUser)})"><label>{{ $t("队员 Table 用户 ID") }}<input v-model="inviteUser" type="number" min="1" required /></label><button :disabled="busy">{{ $t("发出邀请") }}</button></form>
                  <div v-for="invite in pendingInvites" :key="invite.user_id" class="invite-row"><span>{{ invite.display_name }} · {{ $t('等待接受邀请') }}</span><button :disabled="busy || !editable" @click="act('cancel_invite',{team_id:myTeam.id,user_id:invite.user_id})">{{ $t("取消邀请") }}</button></div>
                  <p class="enrollment-meta">{{ $t('队长确认可选，不影响举办方锁定名单。') }}</p>
                  <div class="enrollment-actions"><button :disabled="busy || state.roster_locked || (!myTeam.submitted && members(myTeam).length !== state.team_size)" @click="act(myTeam.submitted?'unsubmit_team':'submit_team',{team_id:myTeam.id})">{{ $t(myTeam.submitted ? '撤回队伍确认' : '确认队伍（可选）') }}</button><button class="enrollment-danger" :disabled="busy || !editable" @click="act('disband_team',{team_id:myTeam.id})">{{ $t("解散队伍") }}</button></div>
                </template><button v-else :disabled="busy || !editable" @click="act('leave_team',{team_id:myTeam.id})">{{ $t("离开队伍（保留个人报名）") }}</button>
              </section>
            </template>
          </template>
        </div>
      </details>
      <section class="roster-section">
        <div class="roster-heading"><h3>{{ $t(state.mode==='solo'?'参赛选手':'参赛队伍') }}</h3><label class="roster-search"><span class="sr-only">{{ $t('搜索队名、选手或 ID') }}</span><input v-model="rosterSearch" type="search" :placeholder="$t('搜索队名、选手或 ID')" /></label></div>
        <div class="enrolled-teams">
          <article v-for="team in shownTeams" :key="team.id" class="roster-team">
            <header><h4>{{ team.name }}</h4><span class="team-count" :class="{ complete:members(team).length===state.team_size }">{{ members(team).length }}/{{ state.team_size }}</span></header>
            <div v-for="member in members(team)" :key="member.user_id" class="roster-member"><span class="roster-position">{{ member.position || '—' }}</span><span class="member-name">{{ member.display_name }}<small>ID {{ member.user_id }}</small></span><span class="member-badges"><span v-if="team.captain_user_id===member.user_id" class="captain-badge">{{ $t('队长') }}</span><span v-if="member.is_external" class="guest-badge">{{ $t('外援') }}</span></span></div>
            <p v-if="!members(team).length" class="enrollment-meta">{{ $t('等待队员加入') }}</p>
          </article>
        </div>
        <section v-if="shownUnassigned.length" class="unassigned-roster"><h4 v-if="state.mode!=='solo'">{{ $t('待分组选手') }}</h4><div class="unassigned-list"><div v-for="member in shownUnassigned" :key="member.user_id" class="roster-member"><span class="member-name">{{ member.display_name }}<small>ID {{ member.user_id }}</small></span><span v-if="member.is_external" class="guest-badge">{{ $t('外援') }}</span></div></div></section>
        <p v-if="!state.entries.length" class="roster-empty">{{ $t("暂无报名或导入名单。") }}</p>
        <p v-else-if="!shownTeams.length && !shownUnassigned.length" class="roster-empty">{{ $t('没有匹配的队伍或选手。') }}</p>
      </section>
      <details v-if="state.me.can_manage" ref="managementPanel" class="enrollment-admin">
        <summary><span><strong>{{ $t('赛事名单管理') }}</strong><small>{{ $t('仅举办方与赛事管理员可见') }}</small></span><span class="summary-hint">{{ $t('展开管理') }}</span></summary>
        <div class="management-content">
          <nav class="management-tabs" :aria-label="$t('赛事名单管理')"><button v-for="[key,label] in [['import','导入名单'],['settings','报名设置'],['locks','名单锁定'],['audit','操作记录']]" v-show="key!=='import' || !state.roster_locked" :key="key" type="button" :aria-pressed="managementTab===key" @click="managementTab=key">{{ $t(label) }}</button></nav>
          <section v-if="managementTab==='settings'" class="management-section">
            <h3>{{ $t('报名设置') }}</h3>
            <p class="enrollment-meta">{{ $t('设置报名方式、人数上限与开放状态。') }}</p>
            <form v-if="!state.roster_locked && !state.registration_locked" @submit.prevent="act('settings',{mode,capacity:Number(capacity),registration_open:open})"><label>{{ $t("报名方式") }}<select v-model="mode" :disabled="state.fixed_policy || !!state.entries.length"><option v-for="(label,key) in labels" :key="key" :value="key">{{ $t(label) }}</option></select></label><label>{{ $t("人数上限（0 不限）") }}<input v-model="capacity" type="number" min="0" max="10000" :disabled="state.fixed_policy" /></label><label class="checkbox-label"><input v-model="open" type="checkbox" />{{ $t("开放报名") }}</label><button class="enrollment-primary" :disabled="busy">{{ $t("保存设置") }}</button></form>
            <p v-else class="enrollment-feedback">{{ $t('名单已锁定，调整报名设置需先解锁。') }}</p>
          </section>
          <section v-if="managementTab==='locks'" class="management-section">
            <h3>{{ $t('名单锁定') }}</h3>
            <p class="enrollment-meta">{{ $t('举办方可直接锁定完整名单，无需队长先提交报名。') }}</p>
            <div class="lock-options">
              <article><span class="section-kicker">01</span><h4>{{ $t('锁定参赛人员') }}</h4><p>{{ $t('停止新增报名与退出，仍可调整同一批选手的分组。') }}</p><button :disabled="busy || state.registration_locked" @click="act('lock_registration')">{{ $t(state.registration_locked?'已锁定':'锁定参赛人员') }}</button></article>
              <article><span class="section-kicker">02</span><h4>{{ $t('锁定最终名单') }}</h4><p>{{ $t('确认分组、人数、队长及队内序号，冻结本届参赛阵容。') }}</p><button class="enrollment-primary" :disabled="busy || state.roster_locked" @click="act('lock_roster')">{{ $t(state.roster_locked?'已锁定':'锁定最终名单') }}</button></article>
            </div>
            <form v-if="state.registration_locked || state.roster_locked" class="unlock-form" @submit.prevent="act('unlock',{reason:unlockReason})"><label>{{ $t("解锁原因") }}<input v-model="unlockReason" minlength="4" maxlength="500" required /></label><button :disabled="busy">{{ $t("解锁（不会自动开放报名）") }}</button></form>
          </section>
        <section v-if="managementTab==='import' && !state.roster_locked" class="management-section">
          <h3>{{ $t("导入 / 调整名单") }}</h3>
          <div class="roster-import-controls">
            <label>{{ $t("名单格式") }}<select v-model="importFormat" :disabled="busy" @change="preview=null"><option value="teams" :disabled="state.mode==='solo'">{{ $t("按队伍导入") }}</option><option value="advanced">{{ $t("高级导入（逐人）") }}</option></select></label>
            <label>{{ $t("账号匹配方式") }}<select v-model="identityMode" :disabled="busy" @change="preview=null"><option value="username">{{ $t("当前用户名") }}</option><option value="user_id">{{ $t("用户 ID") }}</option></select></label>
            <button :disabled="busy" @click="exportToEditor">{{ $t("载入当前名单编辑") }}</button>
          </div>
          <p v-if="importFormat==='teams'">{{ $t('每行：队名、队长、其余队员。支持 Excel 粘贴或逗号分隔，自动生成队内序号。') }} {{ $t(`每队 ${state.team_size} 人。`) }}</p>
          <p v-else>{{ $t("每行一人：用户名或 ID、队名（可空）、外援0或1、队长0或1、队内序号（可空）。也可仅填写用户名或 ID。") }}</p>
          <label>{{ $t("名单") }}<textarea v-model="csv" rows="6" :disabled="busy" :placeholder="importPlaceholder" @input="preview=null" /></label>
          <div class="import-footer"><p class="enrollment-meta">{{ $t('账号匹配失败则整批不导入；保存会替换现有名单并取消旧邀请。') }}</p><button class="enrollment-primary" :disabled="busy || !csv.trim()" @click="importList(false)">{{ $t("校验并预览") }}</button></div>
          <details class="import-help"><summary>{{ $t('导入规则与注意事项') }}</summary><p>{{ $t('首位默认队长，队内序号按列顺序生成；外援默认关闭，均可在预览调整。') }}</p><p>{{ $t('用户名仅匹配当前有效账号。人员锁定后只能调整原有人员的分组。完整名单可由举办方直接锁定，无需队长再次提交。') }}</p></details>
          <div v-if="preview" class="roster-preview">
            <h3>{{ $t("预览 · 尚未保存") }}</h3>
            <p class="enrollment-meta">{{ $t("改选队长会将该选手调整为 1 号位，并交换原 1 号位的序号。确认保存前请核对队长与外援。") }}</p>
            <div class="roster-preview-teams">
              <article v-for="group in previewTeams" :key="group.name" class="roster-preview-team">
                <h4>{{ group.name || $t('未分组') }}</h4>
                <label v-if="group.name">{{ $t('队长') }}<select :value="group.entries.find(entry=>entry.captain)?.user_id || ''" :disabled="busy" @change="selectRosterCaptain(preview.entries,Number($event.target.value))"><option value="" disabled>{{ $t('请选择队长') }}</option><option v-for="entry in group.entries" :key="entry.user_id" :value="entry.user_id">{{ entry.display_name }}</option></select></label>
                <div v-for="entry in group.entries" :key="entry.user_id" class="roster-preview-player">
                  <span class="roster-position">{{ entry.position || '—' }}</span>
                  <span>{{ entry.display_name }}<small>ID {{ entry.user_id }}</small></span>
                  <label class="checkbox-label"><input v-model="entry.is_external" type="checkbox" :disabled="busy" />{{ $t('外援') }}</label>
                </div>
              </article>
            </div>
            <button class="enrollment-primary" :disabled="busy" @click="importList(true)">{{ $t("确认替换为这 ") }}{{ $t(preview.entries.length) }}{{ $t(" 位选手") }}</button>
          </div>
        </section>
          <section v-if="managementTab==='audit'" class="management-section"><h3>{{ $t("最近操作记录") }}</h3><p v-if="!state.audit.length" class="enrollment-meta">{{ $t('暂无操作记录。') }}</p><div v-for="item in state.audit" :key="item.revision" class="audit-row"><strong>{{ $t(item.action) }}</strong><span>#{{ item.revision }} · ID {{ item.actor_user_id }}</span><time>{{ item.created_at }}</time><p v-if="item.action==='unlock'">{{ JSON.parse(item.payload_json).reason }}</p></div></section>
        </div>
      </details>
    </template>
  </section>
</template>

<style scoped>
.enrollment-panel{margin:16px 0 28px;font-size:14px;color:var(--competition-text)}
.enrollment-panel h2,.enrollment-panel h3,.enrollment-panel h4{margin:0;line-height:1.4}
.enrollment-panel h2{font-size:24px;letter-spacing:-.025em}.enrollment-panel h3{font-size:18px}.enrollment-panel h4{font-size:16px}
.enrollment-panel p{line-height:1.65}.enrollment-panel small,.enrollment-meta{color:var(--competition-muted);font-size:12px}
.enrollment-panel a{color:var(--competition-accent)}
.enrollment-panel button{min-height:38px;padding:8px 13px;border:1px solid var(--competition-border);border-radius:7px;background:var(--competition-card);color:inherit;font:inherit;font-size:13px;cursor:pointer}
.enrollment-panel button:hover:not(:disabled){border-color:var(--competition-accent)}.enrollment-panel button:disabled{opacity:.45;cursor:default}
.enrollment-panel .enrollment-primary{background:var(--competition-accent);border-color:var(--competition-accent);color:var(--competition-page);font-weight:700}
.enrollment-panel .enrollment-danger{color:light-dark(#a9443c,#fda4af)}
.enrollment-panel :is(button,input,select,textarea,summary):focus-visible{outline:2px solid var(--competition-accent);outline-offset:3px}
.enrollment-panel label{display:grid;gap:7px;min-width:0;font-size:12px;color:var(--competition-muted)}
.enrollment-panel :is(input,select,textarea){max-width:100%;padding:10px 12px;border:1px solid var(--competition-border);border-radius:7px;background:var(--competition-page);color:var(--competition-text);font:inherit;font-size:14px;box-sizing:border-box}
.enrollment-panel input[type=checkbox]{accent-color:var(--competition-accent);width:16px;height:16px;margin:0;flex:none}
.enrollment-panel form,.enrollment-actions,.roster-import-controls{display:flex;align-items:end;flex-wrap:wrap;gap:10px;margin:12px 0}
.enrollment-panel .checkbox-label{display:flex;align-items:center;gap:7px;min-height:38px;white-space:nowrap}
.enrollment-heading,.heading-actions,.roster-heading{display:flex;align-items:center;justify-content:space-between;gap:16px}
.heading-actions{justify-content:flex-end;flex-wrap:wrap;gap:10px}
.section-kicker{color:var(--competition-accent);font-size:11px;font-weight:700;letter-spacing:.12em;margin:0 0 7px}
.status-badge,.team-count{padding:5px 9px;border:1px solid var(--competition-border);border-radius:20px;color:var(--competition-muted);font-size:11px;white-space:nowrap}
.status-badge.active,.team-count.complete{color:light-dark(#387957,#8fceb1);background:color-mix(in srgb,#54a879 8%,transparent);border-color:color-mix(in srgb,#54a879 22%,transparent)}
.enrollment-overview{display:flex;align-items:center;gap:24px;margin:16px 0;padding:12px 16px;border:1px solid var(--competition-border);border-radius:10px;background:var(--competition-card)}
.enrollment-overview>div{display:flex;align-items:baseline;gap:8px;min-width:78px}.enrollment-overview strong{font-size:24px;line-height:1.2;font-variant-numeric:tabular-nums}.enrollment-overview strong small{font-size:13px;font-weight:400}.enrollment-overview span{color:var(--competition-muted);font-size:12px}
.enrollment-overview>p{display:grid;gap:4px;margin:0 0 0 auto;padding-left:24px;border-left:1px solid var(--competition-border);font-size:13px}
.personal-registration,.enrollment-admin{border:1px solid var(--competition-border);border-radius:12px;background:var(--competition-card);overflow:hidden}
.personal-registration>summary,.enrollment-admin>summary{display:flex;justify-content:space-between;align-items:center;gap:12px;list-style:none;padding:12px 16px;cursor:pointer}
.personal-registration>summary::-webkit-details-marker,.enrollment-admin>summary::-webkit-details-marker{display:none}
.personal-registration>summary>span:first-child,.enrollment-admin>summary>span:first-child{display:flex;align-items:baseline;flex-wrap:wrap;gap:12px}
.summary-hint{color:var(--competition-accent);font-size:12px;white-space:nowrap}.summary-hint::after{content:' +';font-size:16px}details[open]>summary .summary-hint::after{content:' −'}
.personal-content,.management-content{padding:0 20px 20px}.personal-content{border-top:1px solid var(--competition-border);padding-top:16px}.personal-content p:last-child{margin-bottom:0}
.invite-row{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;padding:12px 0;border-top:1px solid var(--competition-border)}.invite-row .enrollment-actions{margin:0}
.roster-section{margin:20px 0}.roster-heading{margin-bottom:12px}.roster-search{max-width:260px;width:45%}.roster-search input{width:100%;background:var(--competition-card)}
.enrolled-teams,.roster-preview-teams{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:12px}
.roster-team{min-width:0;padding:12px 14px;border:1px solid var(--competition-border);border-radius:10px;background:var(--competition-card)}.roster-team>header{display:flex;justify-content:space-between;align-items:center;gap:10px;padding-bottom:9px;border-bottom:1px solid var(--competition-border)}.roster-team h4{font-size:15px;overflow-wrap:anywhere}
.roster-member,.roster-preview-player{display:flex;align-items:center;gap:8px;min-height:38px;padding:7px 0;border-bottom:1px solid var(--competition-border)}.roster-member:last-child{border:0;padding-bottom:0}
.roster-position{flex:none;display:grid;place-items:center;width:25px;height:25px;border-radius:6px;color:var(--competition-muted);background:var(--competition-page);font-size:12px;font-variant-numeric:tabular-nums}
.member-name{min-width:0;flex:1;font-size:14px;overflow-wrap:anywhere}.member-name small{display:inline-block;margin-left:7px;font-size:10px;color:var(--competition-muted);white-space:nowrap}.roster-preview-player small{display:inline-block;margin-left:7px;font-size:10px;color:var(--competition-muted)}
.member-badges{display:flex;flex-wrap:wrap;justify-content:end;gap:5px;max-width:90px}.captain-badge,.guest-badge{padding:3px 6px;border-radius:4px;font-size:10px;white-space:nowrap}.captain-badge{color:var(--competition-accent);background:color-mix(in srgb,var(--competition-accent) 10%,transparent)}.guest-badge{color:var(--competition-muted);border:1px solid var(--competition-border)}
.unassigned-roster{margin-top:20px}.unassigned-roster>h4{margin-bottom:10px}.unassigned-list{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:10px}.unassigned-list .roster-member{padding:14px 16px;border:1px solid var(--competition-border);border-radius:8px;background:var(--competition-card)}
.roster-empty{padding:32px;text-align:center;color:var(--competition-muted);border:1px dashed var(--competition-border);border-radius:10px}
.enrollment-admin{margin-top:20px}.management-tabs{display:flex;gap:4px;overflow-x:auto;border-top:1px solid var(--competition-border);border-bottom:1px solid var(--competition-border);margin:0 -20px 16px;padding:0 16px}
.management-tabs button{flex:none;border:0;border-bottom:2px solid transparent;border-radius:0;padding:13px 12px;color:var(--competition-muted);background:none}.management-tabs button[aria-pressed=true]{color:var(--competition-accent);border-bottom-color:var(--competition-accent);font-weight:700}
.management-section>h3{margin-bottom:10px}.management-section>p{max-width:850px;font-size:13px}.management-section>label{margin-top:16px}
.roster-import-controls{margin:12px 0}.roster-import-controls>label{min-width:160px}.roster-import-controls>button{margin-left:auto}
.enrollment-panel textarea{width:100%;resize:vertical;line-height:1.7;min-height:140px;tab-size:4}
.import-footer{display:flex;justify-content:space-between;align-items:center;gap:14px;margin-top:12px}.import-footer p{margin:0;max-width:600px}.import-footer button{flex:none}
.import-help{margin-top:14px;font-size:12px;color:var(--competition-muted)}.import-help summary{cursor:pointer}.import-help p{margin:10px 0 0;max-width:800px}
.roster-preview{margin-top:18px;border-top:1px solid var(--competition-border);padding-top:16px}.roster-preview-teams{margin:12px 0}.roster-preview-team{min-width:0;padding:12px;border:1px solid var(--competition-border);border-radius:10px}.roster-preview-team h4{margin-bottom:10px;overflow-wrap:anywhere}.roster-preview-team select{width:100%}.roster-preview-player>span:nth-child(2){flex:1;min-width:0;overflow-wrap:anywhere}.roster-preview-player .checkbox-label{min-height:0}
.lock-options{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}.lock-options article{padding:20px;border:1px solid var(--competition-border);border-radius:10px;display:flex;flex-direction:column;align-items:start}.lock-options p{color:var(--competition-muted);font-size:13px;flex:1;margin:10px 0 18px}.unlock-form{padding-top:18px;border-top:1px solid var(--competition-border)}.unlock-form label{flex:1}
.audit-row{display:grid;grid-template-columns:1fr auto;gap:6px;padding:14px 0;border-bottom:1px solid var(--competition-border);font-size:12px}.audit-row span,.audit-row time{color:var(--competition-muted)}.audit-row time{grid-column:1/-1;overflow-wrap:anywhere}.audit-row p{margin:0;grid-column:1/-1}
.enrollment-feedback{padding:12px 16px;margin:16px 0;border:1px solid var(--competition-border);border-radius:8px;background:var(--competition-card);font-size:13px;overflow-wrap:anywhere}.enrollment-feedback.error{color:light-dark(#a9443c,#fda4af);border-color:color-mix(in srgb,#c65750 35%,transparent)}.enrollment-feedback.success{color:light-dark(#387957,#8fceb1)}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(max-width:650px){.enrollment-heading{align-items:start}.enrollment-panel h2{font-size:20px}.section-kicker{font-size:10px}.heading-actions{gap:6px;max-width:45%}.status-badge{font-size:10px}.enrollment-overview{padding:12px;gap:12px;flex-wrap:wrap}.enrollment-overview>div{min-width:0;flex:1;display:grid;gap:3px}.enrollment-overview strong{font-size:22px}.enrollment-overview>p{width:100%;border:0;border-top:1px solid var(--competition-border);padding:8px 0 0;margin:0;display:flex;justify-content:space-between;flex-wrap:wrap}.personal-registration>summary,.enrollment-admin>summary{padding:12px 14px}.personal-registration>summary>span:first-child,.enrollment-admin>summary>span:first-child{display:grid;gap:3px}.personal-content,.management-content{padding-inline:14px}.management-tabs{margin-inline:-14px;padding-inline:6px}.management-tabs button{font-size:12px;padding:12px 10px}.enrollment-panel form,.roster-import-controls{align-items:stretch;flex-direction:column}.enrollment-panel form>label,.roster-import-controls>label{width:100%}.roster-import-controls>button{margin-left:0}.roster-heading{flex-wrap:wrap;gap:8px}.roster-search{width:100%;max-width:none}.enrolled-teams{grid-template-columns:1fr}.roster-team{padding:12px 14px}.lock-options{grid-template-columns:1fr}.import-footer{align-items:stretch;flex-direction:column}.import-footer button{align-self:start}.member-badges{max-width:90px}}
@media(max-width:650px){.enrollment-heading{flex-direction:column;align-items:stretch;gap:10px}.heading-actions{max-width:none;justify-content:flex-end;gap:8px}.heading-actions .status-badge{margin-right:auto}.heading-actions button{min-height:34px;padding:6px 10px;font-size:12px}.management-tabs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0;overflow:visible;padding:4px 8px}.management-tabs button{padding:9px 8px;text-align:left}}
</style>
