<script setup>
import { ref, watch } from 'vue';
import { api } from './api.js';
const props = defineProps({ slug: String, teamSize: {type:Number,default:3} });
const emit = defineEmits(['change']);
const enabled = ref(false), roster = ref(null), yellow = ref(''), white = ref(''), start = ref(''), error = ref('');
let requestId = 0;
watch(() => [props.slug, props.teamSize], async ([slug]) => {
  const id = ++requestId;
  enabled.value = false; roster.value = null; yellow.value = ''; white.value = ''; error.value = '';
  if (!slug) return;
  try { const result = await api.enrollment(slug); if(id === requestId) roster.value = result; }
  catch { if(id === requestId) error.value = '无法读取锁定名单，请重新选择赛事。'; }
}, { immediate: true });
watch([enabled, yellow, white, start], () => {
  emit('change', { ...(enabled.value ? { yellow_team_id: yellow.value, white_team_id: white.value } : {}),
    starts_at: start.value ? new Date(start.value).toISOString() : null });
});
</script>
<template>
  <fieldset class="schedule-picker">
    <legend>{{ $t("赛程与锁定名单") }}</legend>
    <p v-if="error" role="alert">{{ $t(error) }}</p>
    <p v-else-if="!roster?.roster_locked">{{ $t("自由房间：任意已登录选手可落座。正式赛程请先锁定报名名单及队内序号。") }}</p>
    <p v-else-if="roster.mode === 'solo' || roster.team_size !== teamSize">{{ $t("赛事名单人数须与房间每队人数一致；统计赛不使用此对战流程。") }}</p>
    <template v-else>
      <label><input v-model="enabled" type="checkbox" />{{ $t("使用锁定队伍创建赛程房间") }}</label>
      <div v-if="enabled" class="schedule-fields">
        <label>{{ $t("黄方队伍") }}<select v-model="yellow" required><option value="" disabled>{{ $t("请选择") }}</option><option v-for="team in roster.teams" :key="team.id" :value="team.id" :disabled="team.id===white">{{ team.name }}</option></select></label>
        <label>{{ $t("白方队伍") }}<select v-model="white" required><option value="" disabled>{{ $t("请选择") }}</option><option v-for="team in roster.teams" :key="team.id" :value="team.id" :disabled="team.id===yellow">{{ team.name }}</option></select></label>
      </div>
    </template>
    <label>{{ $t("预定开战时间（本设备时区）") }}<input v-model="start" type="datetime-local" :required="enabled" /></label>
    <p>{{ $t("未绑定队伍时为自由房间，不计入正式赛事纪录。设置开战时间后，到点方可开始抽签；超过15分钟，已全员落座且队长准备的一方获胜，未就位方判全部对局负，双方均未就位则0:0。") }}</p>
  </fieldset>
</template>
<style scoped>
.schedule-picker{border:1px solid #ded4c5;padding:16px;border-radius:6px}.schedule-picker label{display:flex;gap:8px;align-items:center}.schedule-fields{display:flex;flex-wrap:wrap;gap:16px;margin-top:16px}.schedule-fields label{flex:1;min-width:180px;flex-direction:column;align-items:stretch}.schedule-fields p{width:100%;font-size:13px;line-height:1.7;color:#817567}.schedule-fields select,.schedule-fields input{padding:10px;min-width:0;background:#f5f6f8;border:1px solid #cdd2d8;border-radius:5px;color:inherit}
</style>
<style scoped>
.schedule-picker{border-color:var(--competition-border);color:var(--competition-text)}
.schedule-picker p{color:var(--competition-muted);font-size:13px;line-height:1.7}
.schedule-picker input,.schedule-picker select{background:var(--competition-card);color:var(--competition-text);border:1px solid var(--competition-border);border-radius:5px;padding:8px;max-width:100%}
.schedule-picker input[type=checkbox]{width:auto}
</style>
