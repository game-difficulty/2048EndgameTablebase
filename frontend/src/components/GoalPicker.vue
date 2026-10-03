<template>
  <div class="goal-picker flex min-w-0 items-center gap-1.5">
    <UiSelect :model-value="mode" :options="modes" :aria-label="$t('goals.type')"
      trigger-class="top-menu-select" menu-class="z-[260]" @change="changeMode" />
    <div v-if="mode === 'sum' && editable" class="relative">
      <input ref="sumInput" v-model="draft" type="number" min="4" max="16382" step="2"
        :aria-label="$t('goals.sum')" :aria-invalid="!validSumTarget(draft)"
        class="goal-number" @change="commitSum" @keydown.enter="commitSum" />
      <div class="absolute inset-y-1 right-1 flex w-4 flex-col text-text-secondary">
        <button type="button" :aria-label="$t('tables.increase')" class="flex-1 text-[9px] leading-none hover:text-accent" @click="step(1)">▲</button>
        <button type="button" :aria-label="$t('tables.decrease')" class="flex-1 text-[9px] leading-none hover:text-accent" @click="step(-1)">▼</button>
      </div>
    </div>
    <UiSelect v-else :model-value="modelValue" :options="values" :placeholder="modelValue.replace(/^sum-/, '') || $t('tables.noGoals')"
      :aria-label="$t('goals.value')" trigger-class="top-menu-select" menu-class="z-[260]" @change="select" />
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import UiSelect from './UiSelect.vue';
import { validSumTarget } from '../utils/goalTarget';
const props = defineProps({ modelValue: { type: String, default: '' }, targets: { type: Array, default: () => [] }, editable: Boolean });
const emit = defineEmits(['update:modelValue', 'change']);
const { t } = useI18n();
const mode = ref(props.modelValue.startsWith('sum-') ? 'sum' : 'tile');
const draft = ref(props.modelValue.startsWith('sum-') ? props.modelValue.slice(4) : '1800');
const sumInput = ref(null);
const step = (direction) => {
  if (direction > 0) sumInput.value.stepUp(); else sumInput.value.stepDown();
  draft.value = sumInput.value.value;
  commitSum();
};
const modes = computed(() => [{ value: 'tile', label: t('goals.tile') }, { value: 'sum', label: t('goals.sum') }]);
const values = computed(() => props.targets.map(String).filter(value => value.startsWith('sum-') === (mode.value === 'sum'))
  .map(value => ({ value, label: value.replace(/^sum-/, '') })));
const select = (value) => { emit('update:modelValue', value); emit('change', value); };
const commitSum = () => { select(validSumTarget(draft.value) ? `sum-${Number(draft.value)}` : ''); };
const changeMode = (value) => {
  mode.value = value;
  if (value === 'sum' && props.editable) {
    if (values.value.length) draft.value = values.value[0].label;
    commitSum();
  }
  else select(values.value[0]?.value || '');
};
watch(() => props.modelValue, value => {
  if (!value) return;
  mode.value = value.startsWith('sum-') ? 'sum' : 'tile';
  if (mode.value === 'sum') draft.value = value.slice(4);
});
</script>

<style scoped>
.goal-number { width: 6.5rem; min-width: 0; border: 1px solid var(--border-main); border-radius: .6rem; background: var(--bg-main); color: var(--text-main); padding: .45rem 1.4rem .45rem .65rem; font-size: var(--font-ui-sm); font-weight: 800; outline: none; appearance: textfield; }
.goal-number::-webkit-inner-spin-button, .goal-number::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
.goal-number:focus { border-color: var(--accent); }
.goal-number[aria-invalid="true"] { border-color: #e67c73; }
</style>
