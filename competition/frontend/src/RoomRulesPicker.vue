<script setup>
import { computed, onMounted, ref, watch } from 'vue';
import { language } from './i18n.js';
import { api } from './api.js';
import { ruleSummary, ruleRequest, lineupPolicyDescription } from './roomRules.js';
const props=defineProps({poolSize:Number});
const emit=defineEmits(['change']);
const presets=ref([]), rules=ref(null), error=ref(false), loading=ref(false);
const txt=(zh,en)=>language.value==='zh'?zh:en;
const summary=computed(()=>rules.value?ruleSummary(rules.value,props.poolSize):{valid:false});
function select(preset){rules.value=JSON.parse(JSON.stringify(presets.value.find(p=>p.preset===preset)));}
function custom(){rules.value={...rules.value,preset:'custom'};}
function move(i,offset){const list=rules.value.steps; [list[i],list[i+offset]]=[list[i+offset],list[i]];}
async function load(){loading.value=true; error.value=false;try{presets.value=(await api.roomRulePresets()).presets; select('bo3');}catch{error.value=true;}finally{loading.value=false;}}
onMounted(load);
watch([rules,summary],()=>emit('change',{rules:rules.value?ruleRequest(rules.value):null,valid:summary.value.valid}),{deep:true,immediate:true});
</script>
<template>
  <fieldset class="rules-picker">
    <legend>{{ txt('比赛流程','Match format') }}</legend>
    <p v-if="loading" role="status">{{ txt('正在读取规则模板…','Loading format presets…') }}</p>
    <p v-if="error" role="alert">{{ txt('规则模板读取失败。','Could not load format presets.') }} <button type="button" @click="load">{{ txt('重试','Retry') }}</button></p>
    <template v-if="rules">
      <div class="presets"><button v-for="p in presets" :key="p.preset" type="button" :aria-pressed="rules.preset===p.preset" @click="select(p.preset)"><strong>{{ p.preset.toUpperCase() }}</strong><span>{{ p.game_count }} {{ txt('局 · 至少','games · pool ≥') }} {{ p.minimum_pool_size }} {{ txt('项','') }}</span></button><button type="button" :aria-pressed="rules.preset==='custom'" @click="custom"><strong>{{ txt('自定义','Custom') }}</strong><span>{{ txt('编辑 BP 顺序','Edit BP sequence') }}</span></button></div>
      <div class="rule-fields">
        <label>{{ txt('每队人数','Players per team') }}<input v-model.number="rules.team_size" type="number" min="1" max="16" required /></label>
        <label>{{ txt('比赛结束方式','Series completion') }}<select v-model="rules.series_mode"><option value="all">{{ txt('打满全部对局','Play every game') }}</option><option value="best_of">{{ txt('先达到多数胜场结束','First to majority wins') }}</option></select></label>
        <label>{{ txt('选手出场限制','Lineup policy') }}<select v-model="rules.lineup_policy"><option value="unique">{{ txt('每人最多一局','At most once per player') }}</option><option value="everyone">{{ txt('可重复 · 尽可能全员出场','Repeats · maximize participation') }}</option><option value="balanced">{{ txt('可重复 · 尽可能平均出场','Repeats · balanced appearances') }}</option><option value="free">{{ txt('可重复 · 完全不限','Repeats · unrestricted') }}</option></select></label>
        <label>{{ txt('最后一局的选定方式','Final game selection') }}<select v-model="rules.final_selection"><option value="blind">{{ txt('双方盲选后抽签','Draw between two blind picks') }}</option><option value="random">{{ txt('剩余项目池随机抽取','Random from remaining pool') }}</option></select></label>
      </div>
      <p class="rule-note" aria-live="polite">{{ lineupPolicyDescription(rules.lineup_policy, language) }} {{ txt('例如 3 人打 7 局：全员出场允许 5/1/1，平均出场只允许 3/2/2 的排列。','For 3 players over 7 games: participation allows 5/1/1; balance requires a permutation of 3/2/2.') }}</p>
      <p v-if="rules.series_mode==='best_of'" class="rule-note">{{ txt('出场限制按完整布阵校验；提前结束时，实际出场人数和次数可能不满足完整布阵的分配。若要保证实际分配，请选择打满全部对局。','Policies apply to the full planned lineup. An early finish may leave actual participation or counts below that plan. Choose “Play every game” to ensure the full allocation is played.') }}</p>
      <ol class="bp-preview"><li v-for="(step,i) in rules.steps" :key="i"><b>{{ String(i+1).padStart(2,'0') }}</b><span>{{ step.actor==='first'?txt('先手','First'):txt('后手','Second') }}</span><strong>{{ step.bans?`B${step.bans} `:'' }}{{ step.picks?`P${step.picks}`:'' }}</strong></li><li><b>★</b><span>{{ txt('抽签','Draw') }}</span><strong>P1</strong></li></ol>
      <div v-if="rules.preset==='custom'" class="step-editor">
        <div v-for="(step,i) in rules.steps" :key="i" class="step-row"><span>{{ i+1 }}</span><select v-model="step.actor" :aria-label="txt('行动方','Acting side')"><option value="first">{{ txt('先手','First') }}</option><option value="second">{{ txt('后手','Second') }}</option></select><label>B <input v-model.number="step.bans" type="number" min="0" max="15" required /></label><label>P <input v-model.number="step.picks" type="number" min="0" max="15" required /></label><button type="button" :disabled="!i" :aria-label="txt('上移步骤','Move step up')" @click="move(i,-1)">↑</button><button type="button" :disabled="i===rules.steps.length-1" :aria-label="txt('下移步骤','Move step down')" @click="move(i,1)">↓</button><button type="button" :disabled="rules.steps.length===1" :aria-label="txt('删除步骤','Remove step')" @click="rules.steps.splice(i,1)">×</button></div>
        <button type="button" :disabled="rules.steps.length>=24" @click="rules.steps.push({actor:'first',bans:0,picks:1})">+ {{ txt('添加步骤','Add step') }}</button>
      </div>
      <p class="rule-note">{{ txt('B = 禁用，P = 选择。同一步的选禁一起提交；选择顺序即对局顺序。超时按项目池顺序自动选择。','B = ban, P = pick. Submit a step together; pick order determines game order. Timeouts use pool order.') }}</p>
      <details><summary>{{ txt('计时设置','Timers') }}</summary><div class="rule-fields"><label>{{ txt('每步 BP（秒）','Per BP step (seconds)') }}<input v-model.number="rules.draft_seconds" type="number" min="5" max="3600" required /></label><label>{{ txt('布阵（秒）','Lineup (seconds)') }}<input v-model.number="rules.lineup_seconds" type="number" min="5" max="3600" required /></label><label>{{ txt('每队总计时（秒）','Team clock (seconds)') }}<input v-model.number="rules.team_clock_seconds" type="number" min="30" max="86400" required /></label></div></details>
      <div class="rule-total" aria-live="polite"><strong>{{ summary.games }} {{ txt('局','games') }} · {{ rules.team_size }} v {{ rules.team_size }}</strong><span>{{ txt('项目池至少','Pool minimum:') }} {{ summary.minimum }} / {{ props.poolSize }} {{ txt('已选','selected') }}</span></div>
      <p v-if="!summary.valid" class="rule-error" role="alert">{{ txt('请检查人数、计时及项目池。总局数须为 1–15 的奇数；不重复出场时人数不能少于局数。','Check team size, timers and pool. Games must be odd (1–15); unique lineups need at least one player per game.') }}</p>
      <p class="rule-note">{{ rules.series_mode==='best_of'?txt(`先赢 ${Math.floor(summary.games/2)+1} 局结束；平局不计胜场，打满仍可平局。`,`First to ${Math.floor(summary.games/2)+1} wins; draws do not count as wins. The series can draw at the game limit.`):txt('即使已确定胜负，也会打完全部对局。','Every game is played, even if the winner is already decided.') }} {{ txt('创建后规则固定，不影响其他房间。','Rules are frozen on creation and do not affect other rooms.') }}</p>
    </template>
  </fieldset>
</template>
<style scoped>
.rules-picker{min-width:0;padding:20px;border:1px solid var(--competition-border);border-radius:12px;display:grid;gap:16px}.rules-picker legend{font-size:18px;font-weight:700;padding:0 8px}.presets{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}.presets button{display:grid;text-align:left;gap:6px;padding:16px}.presets strong{font-size:20px}.presets span,.rule-note{font-size:13px;color:var(--competition-muted);line-height:1.6}.rules-picker button,.rules-picker input,.rules-picker select{font:inherit;color:inherit;background:var(--competition-card);border:1px solid var(--competition-border);border-radius:7px;padding:9px;min-width:0}.presets button[aria-pressed=true]{border-color:var(--competition-accent);box-shadow:inset 0 0 0 1px var(--competition-accent)}.rule-fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.rule-fields label{display:grid;gap:7px;font-size:14px}.bp-preview{list-style:none;display:flex;flex-wrap:wrap;gap:8px;padding:0;margin:0}.bp-preview li{display:flex;gap:10px;align-items:center;border:1px solid var(--competition-border);border-radius:6px;padding:10px;font-size:13px}.bp-preview b{color:var(--competition-accent)}.step-editor{display:grid;gap:8px}.step-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.step-row label{display:flex;align-items:center;gap:6px}.step-row input{width:60px}.rule-total{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;border-top:1px solid var(--competition-border);padding-top:16px}.rule-note{margin:0}.rule-error{color:#c35f49}.rules-picker summary{cursor:pointer;margin-bottom:12px}.rules-picker button{cursor:pointer}.rules-picker button:disabled{opacity:.45;cursor:default}@media(max-width:600px){.presets{grid-template-columns:repeat(2,minmax(0,1fr))}.rule-fields{grid-template-columns:1fr}.rules-picker{padding:14px}}
</style>
