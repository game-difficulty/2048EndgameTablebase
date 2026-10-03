<template>
  <span class="native-goal-picker">
    <select v-if="modes.length > 1" :value="kind" :aria-label="zh ? '目标类型' : 'Goal type'" @change="changeKind($event.target.value)">
      <option v-for="mode in modes" :key="mode" :value="mode">{{ mode === 'sum' ? (zh ? '盘面和' : 'Board sum') : (zh ? '目标方块' : 'Tile') }}</option>
    </select>
    <span v-else-if="kind === 'sum'">{{ zh ? '盘面和' : 'Board sum' }}</span>
    <select :value="modelValue" :aria-label="zh ? '目标数值' : 'Goal value'" @change="$emit('update:modelValue', $event.target.value)">
      <option v-for="value in values" :key="value" :value="value">{{ parseGoalTarget(value)?.value || value }}</option>
    </select>
  </span>
</template>
<script setup>
import { computed } from 'vue';
import { parseGoalTarget, compareGoalTargets } from '../utils/goalTarget.js';
const props = defineProps({ modelValue: String, options: { type: Array, default: () => [] }, language: { type: String, default: 'en' } });
const emit = defineEmits(['update:modelValue']);
const zh = computed(() => props.language.startsWith('zh'));
const kind = computed(() => parseGoalTarget(props.modelValue)?.kind || parseGoalTarget(props.options[0])?.kind || 'tile');
const modes = computed(() => ['tile', 'sum'].filter(mode => props.options.some(value => parseGoalTarget(value)?.kind === mode)));
const values = computed(() => props.options.filter(value => parseGoalTarget(value)?.kind === kind.value).slice().sort(compareGoalTargets));
function changeKind(mode) {
  const value = props.options.find(item => parseGoalTarget(item)?.kind === mode);
  if (value) emit('update:modelValue', value);
}
</script>
<style scoped>
.native-goal-picker { display: flex; gap: .4rem; align-items: center; min-width: 0; }
.native-goal-picker select { min-width: 0; flex: 1; }
.native-goal-picker > span { font-size: .8em; white-space: nowrap; }
</style>
