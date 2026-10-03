<template>
  <div class="flex min-w-0 items-center gap-1.5">
    <UiSelect v-if="modes.length > 1" :model-value="kind" :options="modes" :aria-label="zh ? '目标类型' : 'Goal type'"
      :trigger-class="triggerClass" :option-class="optionClass" :menu-class="menuClass" :align="align" :disabled="disabled" @update:model-value="selectKind" />
    <span v-else-if="kind === 'sum'" class="shrink-0 text-text-secondary text-xs font-bold">{{ zh ? '盘面和' : 'Board sum' }}</span>
    <UiSelect class="min-w-0 flex-1" :model-value="modelValue" :options="values" :placeholder="placeholder" :aria-label="ariaLabel || (zh ? '目标数值' : 'Goal value')"
      :trigger-class="triggerClass" :option-class="optionClass" :menu-class="menuClass" :align="align" :disabled="disabled" @update:model-value="selectValue" />
  </div>
</template>
<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import UiSelect from './UiSelect.vue';
import { parseGoalTarget } from '../utils/goalTarget.js';
const props = defineProps({
  modelValue: { type: String, default: '' }, options: { type: Array, default: () => [] },
  placeholder: String, ariaLabel: String, triggerClass: String, optionClass: String, menuClass: String,
  align: { type: String, default: 'left' }, disabled: Boolean,
});
const emit = defineEmits(['update:modelValue', 'change']);
const { locale } = useI18n();
const zh = computed(() => String(locale.value).startsWith('zh'));
const normalized = computed(() => props.options.map(option => typeof option === 'object' ? option : { value: String(option), label: String(option) }));
const kind = computed(() => parseGoalTarget(props.modelValue)?.kind || parseGoalTarget(normalized.value[0]?.value)?.kind || 'tile');
const modes = computed(() => ['tile', 'sum'].filter(type => normalized.value.some(option => parseGoalTarget(option.value)?.kind === type))
  .map(value => ({ value, label: value === 'sum' ? (zh.value ? '盘面和' : 'Board sum') : (zh.value ? '目标方块' : 'Tile') })));
const values = computed(() => normalized.value.filter(option => parseGoalTarget(option.value)?.kind === kind.value)
  .map(option => ({ ...option, label: String(parseGoalTarget(option.value).value) })));
function selectValue(value) { emit('update:modelValue', value); emit('change', value); }
function selectKind(value) {
  const first = normalized.value.find(option => !option.disabled && parseGoalTarget(option.value)?.kind === value);
  if (first) selectValue(first.value);
}
</script>
