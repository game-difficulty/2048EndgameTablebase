<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { api, commandId } from './api.js';
import { language, t } from './i18n.js';
import { userFacingError } from './errorMessages.js';
import { scheduleTime } from './scheduleDisplay.js';
import { beijingInput, beijingISO, fixtureStatus, stageGroups } from './fixtureSchedule.js';
import RoomRulesPicker from './RoomRulesPicker.vue';

const props = defineProps({ slug: String, active: Boolean, manageMode: Boolean, rosterRevision: Number });
const emit = defineEmits(['navigate', 'changed', 'manage']);
const txt = (zh, en) => language.value === 'zh' ? zh : en;
const state = ref(null), busy = ref(false), loading = ref(false), error = ref(''), notice = ref('');
const filter = ref('all'), editor = ref(null), editError = ref('');
const stageName = ref(''), groupNames = ref(['A', 'B']), assignments = ref({});
const selectedProjects = ref([]), roomRules = ref(null), rulesValid = ref(false);
let epoch = 0, timer;
const groups = computed(() => stageGroups(state.value?.teams || [], assignments.value, groupNames.value));
const validGroups = computed(() => groups.value.every(g => g.name && g.team_ids.length >= 2 && g.team_ids.length <= 16)
  && new Set(groups.value.map(g => g.name.toLocaleLowerCase())).size === groups.value.length
  && groups.value.reduce((n, g) => n + g.team_ids.length, 0) === (state.value?.teams.length || 0));
const stageValid = computed(() => state.value?.roster_locked && !state.value?.event_finished && validGroups.value
  && stageName.value.trim().length >= 2 && rulesValid.value && roomRules.value?.team_size === state.value?.team_size);
const filterGroups = computed(() => [...new Set((state.value?.stages || []).flatMap(s => s.groups.map(g => g.name)))]);
const totalFixtures = computed(() => (state.value?.stages || []).reduce((n, s) => n + s.fixtures.length, 0));
const scheduledFixtures = computed(() => (state.value?.stages || []).reduce((n, s) => n + s.fixtures.filter(f => f.starts_at).length, 0));
function rows(stage) { return stage.fixtures.filter(f => filter.value === 'all' || (filter.value === 'mine' ? f.is_mine : `group:${f.group_name}` === filter.value)); }
function apply(payload) {
  state.value = payload.schedule;
  if (!selectedProjects.value.length) selectedProjects.value = state.value.projects.map(p => p.project_ref);
}
async function load(background = false) {
  if (busy.value) return;
  const id = epoch, slug = props.slug;
  if (!background) { loading.value = true; error.value = ''; }
  try { const payload = await api.eventFixtures(slug); if (epoch === id && props.slug === slug) apply(payload); }
  catch (cause) { if (epoch === id) error.value = userFacingError(cause); }
  finally { if (epoch === id) loading.value = false; }
}
watch(() => props.slug, () => {
  epoch++; busy.value = false; loading.value = false; state.value = null; editor.value = null; error.value = ''; notice.value = ''; filter.value = 'all';
  assignments.value = {}; selectedProjects.value = []; stageName.value = ''; groupNames.value = ['A', 'B'];
  if (props.slug) load();
}, { immediate: true });
watch(() => [props.active, props.rosterRevision], () => { if (props.active && props.slug) load(true); });
onMounted(() => { timer = setInterval(() => { if (props.active && props.slug && !document.hidden) load(true); }, 15000); });
onBeforeUnmount(() => { epoch++; clearInterval(timer); });
function begin(item, action = 'schedule') {
  editor.value = { id: item.id, revision: item.revision, action, starts_at: beijingInput(item.proposed_at || item.starts_at), room_code: '' };
  editError.value = ''; notice.value = '';
}
async function act(item, action, extra = {}, revision = item.revision) {
  if (busy.value) return;
  busy.value = true; editError.value = ''; error.value = ''; notice.value = '';
  const id = epoch, slug = props.slug;
  try {
    const payload = await api.fixtureAction(slug, item.id, { action, revision, command_id: commandId(), ...extra });
    if (id !== epoch) return;
    apply(payload); editor.value = null; emit('changed');
    notice.value = action === 'schedule' && item.starts_at && !state.value.can_manage
      ? txt('改期已提出，原时间保持有效，等待对方队长确认。', 'Reschedule proposed. The original time remains in effect until the other captain confirms.')
      : action === 'schedule' && !item.starts_at
        ? txt('时间已登记，房间已自动创建，双方可提前进入准备。', 'Time saved and room created. Both teams can join and prepare now.')
        : action === 'confirm' || (action === 'schedule' && item.starts_at)
          ? txt('新时间已生效，双方队长需要重新点击准备。', 'New time applied. Both captains must ready up again.')
          : txt('时刻表已更新。', 'Schedule updated.');
    if (payload.warnings?.length) notice.value += ' ' + txt('相邻场次间隔不足一小时，请确认前一场能够结束。', 'Adjacent matches are less than an hour apart. Ensure the previous match can finish.');
  } catch (cause) {
    if (id !== epoch) return;
    if (editor.value?.id === item.id) editError.value = userFacingError(cause);
    else error.value = userFacingError(cause);
  } finally { if (id === epoch) busy.value = false; }
}
function save(item) {
  if (editor.value.action === 'bind') {
    return act(item, 'bind', { room_code: editor.value.room_code.trim().toUpperCase() }, editor.value.revision);
  }
  const start = beijingISO(editor.value.starts_at);
  if (!start || new Date(start) <= new Date()) {
    editError.value = txt('请选择未来的开赛时间（北京时间）。', 'Choose a future start time (Beijing time).'); return;
  }
  return act(item, 'schedule', { starts_at: start }, editor.value.revision);
}
async function createStage() {
  if (busy.value || !stageValid.value) return;
  busy.value = true; error.value = ''; notice.value = '';
  const id = epoch, slug = props.slug;
  try {
    const payload = await api.createFixtureStage(slug, { name: stageName.value.trim(), groups: groups.value,
      projects: selectedProjects.value, rules: roomRules.value, command_id: commandId() });
    if (id !== epoch) return;
    apply(payload); stageName.value = ''; emit('changed');
    notice.value = txt('小组对阵已生成，尚未预约的对阵不会创建空房间。', 'Fixtures generated. Rooms are only created when a time is booked.');
  } catch (cause) { if (id === epoch) error.value = userFacingError(cause); }
  finally { if (id === epoch) busy.value = false; }
}
</script>

<template>
  <section class="fixture-schedule" :aria-label="txt('小组赛时刻表','Group match schedule')">
    <header class="fixture-heading"><div><h2>{{ txt('小组赛时刻表','Group match schedule') }}</h2><p>{{ txt('双方约定后，任一队长登记时间即可自动建房。','Once both teams agree, either captain can book the time and create the room.') }}</p></div>
      <span v-if="state?.stages.length" class="fixture-count">{{ scheduledFixtures }} / {{ totalFixtures }} {{ txt('已约时间','scheduled') }}</span>
      <button v-if="state?.can_manage && !manageMode" type="button" @click="emit('manage')">{{ txt('设置小组赛','Set up groups') }}</button>
    </header>
    <p v-if="error" class="alert" role="alert">{{ t(error) }} <button type="button" :disabled="busy" @click="load()">{{ txt('刷新','Refresh') }}</button></p>
    <p v-if="notice" class="fixture-notice" role="status">{{ notice }}</p>
    <p v-if="loading && !state" role="status">{{ txt('正在加载时刻表…','Loading schedule…') }}</p>
    <template v-if="state">
      <p class="fixture-help">{{ txt('时间统一为北京时间（UTC+8）。双方全部入座且队长准备后，到点自动进入抽签 / BP；改期需对方队长确认，生效后双方重新准备。未就位沿用房间的迟到规则（当前为 15 分钟）。','All times use Beijing time (UTC+8). With all players seated and both captains ready, the draw / draft starts automatically at the agreed time. Reschedules need the other captain’s confirmation and renewed readiness. Room lateness rules apply (currently 15 minutes).') }}</p>
      <p v-if="!state.stages.length" class="fixture-empty">{{ txt('举办方尚未生成小组对阵。','The organizer has not generated group fixtures yet.') }}</p>
      <nav v-if="state.stages.length" class="fixture-filters" :aria-label="txt('筛选对阵','Filter fixtures')">
        <button v-for="[key,label] in [['all',txt('全部','All')],['mine',txt('我的对战','My matches')],...filterGroups.map(g=>[`group:${g}`,`${g} ${txt('组','group')}`])]" :key="key" type="button" :aria-pressed="filter===key" @click="filter=key">{{ label }}</button>
      </nav>
      <section v-for="stage in state.stages" :key="stage.id" class="fixture-stage">
        <header><h3>{{ stage.name }}</h3><small>{{ txt('新建房间','New rooms') }}: {{ stage.rules.game_count }} {{ txt('局','games') }} · {{ stage.rules.team_size }} v {{ stage.rules.team_size }} · {{ txt('单循环','Round robin') }}</small></header>
        <div v-for="item in rows(stage)" :key="item.id" class="fixture-row" :class="{mine:item.is_mine}">
          <div class="fixture-match"><small>{{ item.group_name }} {{ txt('组','group') }} · {{ txt('第','Round ') }}{{ item.round_number }}{{ txt('轮','') }} · {{ item.game_count }} {{ txt('局','games') }}</small><strong>{{ item.yellow_name }} <span>vs</span> {{ item.white_name }}</strong></div>
          <div class="fixture-time"><time>{{ scheduleTime(item.starts_at) || txt('时间待定','Time TBD') }}</time><small v-if="item.proposed_at">{{ txt('拟改为','Proposed:') }} {{ scheduleTime(item.proposed_at) }}</small></div>
          <div class="fixture-result"><b v-if="item.series_score.yellow!==undefined">{{ item.series_score.yellow }}:{{ item.series_score.white }}</b><span>{{ fixtureStatus(item,language) }}</span></div>
          <div class="fixture-actions">
            <a v-if="item.room_code" :href="`/rooms/${item.room_code}`" @click.prevent="emit('navigate',`/rooms/${item.room_code}`)">{{ txt('进入房间','Enter room') }} →</a>
            <button v-if="item.can_schedule && !item.proposed_at" type="button" :disabled="busy" @click="begin(item)">{{ item.starts_at?txt('改期','Reschedule'):txt('填写时间','Book time') }}</button>
            <button v-if="item.can_confirm" type="button" :disabled="busy" @click="act(item,'confirm')">{{ txt('确认改期','Confirm') }}</button>
            <button v-if="item.can_confirm" type="button" :disabled="busy" @click="act(item,'reject')">{{ txt('拒绝','Reject') }}</button>
            <button v-if="item.can_withdraw" type="button" :disabled="busy" @click="act(item,'cancel_proposal')">{{ txt('撤回改期','Withdraw') }}</button>
            <button v-if="item.can_bind && manageMode" type="button" :disabled="busy" @click="begin(item,'bind')">{{ txt('绑定已有房间','Bind existing room') }}</button>
          </div>
          <form v-if="editor?.id===item.id" class="fixture-editor" @submit.prevent="save(item)">
            <label v-if="editor.action==='schedule'">{{ txt('开赛时间 · 北京时间','Start time · Beijing time') }}<input v-model="editor.starts_at" type="datetime-local" required :disabled="busy" /></label>
            <label v-else>{{ txt('已有正式房间码','Existing official room code') }}<input v-model="editor.room_code" required minlength="4" maxlength="12" :disabled="busy" /></label>
            <button type="submit" :disabled="busy">{{ editor.action==='bind'?txt('确认绑定','Bind room'):item.starts_at&&!state.can_manage?txt('提出改期','Propose new time'):txt('保存','Save') }}</button><button type="button" :disabled="busy" @click="editor=null">{{ txt('取消','Cancel') }}</button>
            <p v-if="editError" class="alert" role="alert">{{ t(editError) }} <button type="button" :disabled="busy" @click="editor=null;load()">{{ txt('重新读取对阵','Reload fixture') }}</button></p>
          </form>
        </div>
        <p v-if="!rows(stage).length" class="fixture-empty">{{ txt('此筛选下暂无对阵。','No matches for this filter.') }}</p>
      </section>
      <details v-if="state.can_manage && manageMode" class="fixture-setup" :open="!state.stages.length">
        <summary>{{ txt('生成新的单循环阶段','Generate a round-robin stage') }}</summary>
        <p>{{ txt('分组与房间规则生成后固定；尚未约定时间的对阵不创建房间。已有房间请在对应对阵上绑定。','Groups and format are frozen on generation. Unbooked fixtures do not create rooms. Bind existing rooms to their corresponding fixtures.') }}</p>
        <p v-if="!state.roster_locked" class="alert">{{ txt('请先在报名与队伍页锁定最终参赛名单。','First lock the final roster under Registration & teams.') }}</p>
        <form @submit.prevent="createStage">
          <label>{{ txt('阶段名称','Stage name') }}<input v-model="stageName" :placeholder="txt('例如：小组赛','For example: Group stage')" required minlength="2" maxlength="60" /></label>
          <div class="fixture-group-names"><label v-for="(_,i) in groupNames" :key="i">{{ txt('小组','Group') }} {{ i+1 }}<input v-model="groupNames[i]" required maxlength="40" /></label><button type="button" :disabled="groupNames.length>=16" @click="groupNames.push(String.fromCharCode(65+groupNames.length))">{{ txt('增加一组','Add group') }}</button><button v-if="groupNames.length>1" type="button" @click="groupNames.pop()">{{ txt('减少一组','Remove last group') }}</button></div>
          <div class="fixture-team-grid"><label v-for="team in state.teams" :key="team.id"><span>{{ team.name }}</span><select v-model="assignments[team.id]"><option :value="undefined">{{ txt('选择小组','Select group') }}</option><option v-for="(name,i) in groupNames" :key="i" :value="i">{{ name }}</option></select></label></div>
          <p class="fixture-help">{{ groups.map(g=>`${g.name}: ${g.team_ids.length} ${txt('队','teams')}`).join(' · ') }} · {{ txt('每组至少两队，每队只能属于一组。','At least two teams per group; each team belongs to one group.') }}</p>
          <fieldset class="fixture-projects"><legend>{{ txt('统一项目池','Shared project pool') }}</legend><label v-for="project in state.projects" :key="project.project_ref"><input v-model="selectedProjects" type="checkbox" :value="project.project_ref" />{{ t(project.name) }}</label></fieldset>
          <RoomRulesPicker :pool-size="selectedProjects.length" @change="({rules,valid})=>{roomRules=rules;rulesValid=valid;}" />
          <p v-if="roomRules && roomRules.team_size!==state.team_size" class="alert">{{ txt('每队人数必须与本赛事名单一致。','Players per team must match this event’s roster.') }}</p>
          <button type="submit" class="fixture-primary" :disabled="busy || !stageValid">{{ busy?txt('正在保存…','Saving…'):txt('确认分组并生成对阵','Confirm groups & generate fixtures') }}</button>
        </form>
      </details>
    </template>
  </section>
</template>

<style scoped>
.fixture-schedule{padding:22px;border:1px solid var(--competition-border);border-radius:12px;background:var(--competition-card);margin-bottom:20px;font-size:14px}.fixture-heading,.fixture-stage>header{display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap}.fixture-heading h2{font-size:20px;margin:0}.fixture-heading p{margin:7px 0;color:var(--competition-muted)}.fixture-count{font-size:12px;font-variant-numeric:tabular-nums;color:var(--competition-accent)}.fixture-help,.fixture-stage small,.fixture-empty{font-size:12px;color:var(--competition-muted);line-height:1.6}.fixture-help{margin:12px 0 16px}.fixture-filters{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}.fixture-schedule button{font:inherit;font-size:12px;border:1px solid var(--competition-border);border-radius:6px;background:var(--competition-page);color:var(--competition-text);padding:7px 10px;cursor:pointer}.fixture-schedule button:disabled{cursor:default;opacity:.5}.fixture-filters button[aria-pressed=true],button.fixture-primary{background:var(--competition-accent);color:var(--competition-page);border-color:var(--competition-accent)}.fixture-stage{margin-top:16px}.fixture-stage h3{font-size:16px;margin:0 0 8px}.fixture-row{display:grid;grid-template-columns:minmax(180px,1.7fr) minmax(130px,1fr) auto minmax(130px,auto);align-items:center;gap:14px;padding:13px 0;border-top:1px solid var(--competition-border)}.fixture-match,.fixture-time{display:grid;gap:4px;min-width:0}.fixture-match strong{font-size:14px;overflow-wrap:anywhere}.fixture-match strong span{font-weight:400;color:var(--competition-muted);font-size:12px;margin:0 5px}.fixture-time time{font-variant-numeric:tabular-nums}.fixture-result{display:grid;gap:4px;text-align:center;font-size:12px}.fixture-result b{font-size:17px}.fixture-actions{display:flex;align-items:center;justify-content:flex-end;gap:7px;flex-wrap:wrap}.fixture-actions a{color:var(--competition-accent);text-decoration:none;font-size:12px;white-space:nowrap}.fixture-editor{grid-column:1/-1;display:flex;gap:8px;align-items:flex-end;flex-wrap:wrap;padding:12px;border:1px solid var(--competition-border);border-radius:8px;background:var(--competition-page)}.fixture-editor label,.fixture-setup form>label,.fixture-group-names label{display:grid;gap:5px}.fixture-editor .alert{flex-basis:100%;margin:4px 0}.fixture-schedule input:not([type=checkbox]),.fixture-schedule select{box-sizing:border-box;max-width:100%;min-width:0;padding:8px;border:1px solid var(--competition-border);border-radius:5px;background:var(--competition-card);color:var(--competition-text);font:inherit}.fixture-notice{padding:10px 12px;border-left:3px solid var(--competition-accent);background:var(--competition-page);line-height:1.6}.fixture-setup{border-top:1px solid var(--competition-border);padding-top:18px;margin-top:20px}.fixture-setup summary{cursor:pointer;font-weight:600}.fixture-setup>p{font-size:12px;line-height:1.6;color:var(--competition-muted)}.fixture-setup form{display:grid;gap:16px}.fixture-group-names{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end}.fixture-group-names label{max-width:140px}.fixture-team-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px}.fixture-team-grid label{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 10px;border:1px solid var(--competition-border);border-radius:6px}.fixture-team-grid label span{overflow-wrap:anywhere}.fixture-team-grid select{max-width:110px}.fixture-projects{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;border:1px solid var(--competition-border);border-radius:8px;padding:12px}.fixture-projects label{display:flex;align-items:center;gap:6px;font-size:12px}.fixture-schedule :is(button,a,input,select):focus-visible{outline:2px solid var(--competition-accent);outline-offset:3px}@media(max-width:760px){.fixture-schedule{padding:16px}.fixture-row{grid-template-columns:1fr auto;gap:9px}.fixture-match{grid-column:1/-1}.fixture-time{grid-column:1}.fixture-result{grid-column:2}.fixture-actions{grid-column:1/-1;justify-content:flex-start}.fixture-editor label{width:100%}.fixture-editor input{width:100%}.fixture-stage>header small{font-size:11px}.fixture-heading p{font-size:12px;line-height:1.6}}
</style>

<style scoped>
.fixture-projects{min-width:0}.fixture-projects input[type=checkbox]{width:16px;height:16px;flex:0 0 16px;margin:0;accent-color:var(--competition-accent)}
</style>
