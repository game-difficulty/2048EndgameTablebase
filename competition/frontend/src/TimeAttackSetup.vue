<script setup>
import { computed } from 'vue';
import { language } from './i18n.js';
import { publicRoomMode } from './publicRoomModes.js';
const props=defineProps({modelValue:Object});
const emit=defineEmits(['update:modelValue']);
const en=computed(()=>language.value==='en');
function set(key,value){emit('update:modelValue',{...props.modelValue,[key]:value});}
</script>
<template>
  <div class="target-setup">
    <label>{{en?'Board':'棋盘变体'}}<select :value="modelValue.variant" @change="set('variant',$event.target.value)"><option v-for="v in ['4x4','3x4','2x4','3x3']" :value="v" :key="v">{{v.replace('x',' × ')}}</option></select></label>
    <label>{{en?'Target type':'目标类型'}}<select :value="modelValue.target_kind" @change="set('target_kind',$event.target.value)"><option value="tile">{{en?'Target tile (at least)':'目标数字（不小于）'}}</option><option value="board_sum">{{en?'Board sum (exact)':'目标盘面和（精确等于）'}}</option></select></label>
    <label>{{en?'Target value':'具体数值'}}<input :value="modelValue.target_value" type="number" :min="modelValue.target_kind==='tile'?8:10" max="2147483648" :step="modelValue.target_kind==='tile'?1:2" required @input="set('target_value',$event.target.value)" /></label>
    <p v-if="!publicRoomMode('time_attack').valid(modelValue)" class="alert" role="alert">{{en?'Use a power of two ≥ 8, or an even board sum ≥ 10.':'目标数字须为不小于 8 的 2 的幂，盘面和须为不小于 10 的偶数。'}}</p>
  </div>
</template>
<style scoped>
.target-setup{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.target-setup label{display:grid;gap:8px}.target-setup input,.target-setup select{width:100%;min-width:0;min-height:44px;box-sizing:border-box}.target-setup .alert{grid-column:1/-1}@media(max-width:720px){.target-setup{grid-template-columns:1fr}}
</style>
