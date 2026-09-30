<script setup>
import { ref, watch } from 'vue';
import { api } from './api.js';
const props = defineProps({ slug: String });
const emit = defineEmits(['change']);
const enabled = ref(false), roster = ref(null), yellow = ref(''), white = ref(''), start = ref(''), error = ref('');
let requestId = 0;
watch(() => props.slug, async slug => {
  const id = ++requestId;
  enabled.value = false; roster.value = null; yellow.value = ''; white.value = ''; error.value = '';
  if (!slug) return;
  try { const result = await api.enrollment(slug); if(id === requestId) roster.value = result; }
  catch { if(id === requestId) error.value = '无法读取锁定名单，请重新选择赛事。'; }
}, { immediate: true });
watch([enabled, yellow, white, start], () => {
  emit('change', enabled.value ? { yellow_team_id: yellow.value, white_team_id: white.value,
    starts_at: start.value ? new Date(start.value).toISOString() : null } : {});
});
</script>
<template>
  <fieldset v-if="slug" class="schedule-picker">
    <legend>赛程与锁定名单</legend>
    <p v-if="error" role="alert">{{ error }}</p>
    <p v-else-if="!roster?.roster_locked">请先在赛事页锁定最终名单，再编排队伍和开战时间。</p>
    <p v-else-if="roster.mode === 'solo' || roster.team_size !== 3">当前对战房间需要三人团队名单；单人赛和统计赛不使用此对战流程。</p>
    <template v-else>
      <label><input v-model="enabled" type="checkbox" />使用锁定队伍创建赛程房间</label>
      <div v-if="enabled" class="schedule-fields">
        <label>黄方队伍<select v-model="yellow" required><option value="" disabled>请选择</option><option v-for="team in roster.teams" :key="team.id" :value="team.id" :disabled="team.id===white">{{ team.name }}</option></select></label>
        <label>白方队伍<select v-model="white" required><option value="" disabled>请选择</option><option v-for="team in roster.teams" :key="team.id" :value="team.id" :disabled="team.id===yellow">{{ team.name }}</option></select></label>
        <label>预定开战时间（本设备时区）<input v-model="start" type="datetime-local" required /></label>
<p>可提前进入签到与准备，到点后开始抽签。开战时间后超过 10 分钟仍未全员签到的一方判 0:3 负；双方均未到齐则等待管理员处理。创建后固定名单与时间。</p>
      </div>
    </template>
  </fieldset>
</template>
<style scoped>
.schedule-picker{border:1px solid #ded4c5;padding:16px;border-radius:6px}.schedule-picker label{display:flex;gap:8px;align-items:center}.schedule-fields{display:flex;flex-wrap:wrap;gap:16px;margin-top:16px}.schedule-fields label{flex:1;min-width:180px;flex-direction:column;align-items:stretch}.schedule-fields p{width:100%;font-size:13px;line-height:1.7;color:#817567}.schedule-fields select,.schedule-fields input{padding:10px;min-width:0;background:#f5f6f8;border:1px solid #cdd2d8;border-radius:5px;color:inherit}
</style>
