<template>
  <div class="battle-number-control">
    <input
      :value="modelValue"
      type="number"
      :min="min"
      :max="max"
      :step="step"
      :placeholder="placeholder"
      class="battle-number-input tabular-nums"
      @input="handleInput"
      @change="normalizeCurrent"
    />
    <div class="battle-number-steppers">
      <button type="button" :aria-label="$t('battle.actions.increase')" @click="stepValue(1)">
        <span class="battle-number-arrow up" />
      </button>
      <button type="button" :aria-label="$t('battle.actions.decrease')" @click="stepValue(-1)">
        <span class="battle-number-arrow down" />
      </button>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({
  modelValue: { type: [Number, String], default: '' },
  min: { type: Number, default: Number.NEGATIVE_INFINITY },
  max: { type: Number, default: Number.POSITIVE_INFINITY },
  step: { type: Number, default: 1 },
  placeholder: { type: String, default: '' },
  allowEmpty: { type: Boolean, default: false },
});

const emit = defineEmits(['update:modelValue']);

const clamp = (value) => Math.min(props.max, Math.max(props.min, value));
const normalize = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return props.allowEmpty ? '' : clamp(0);
  const origin = Number.isFinite(props.min) ? props.min : 0;
  const step = Math.max(Number(props.step) || 1, Number.EPSILON);
  const stepped = origin + Math.round((numeric - origin) / step) * step;
  return clamp(Number(stepped.toFixed(10)));
};

const handleInput = (event) => {
  const raw = event.target.value;
  emit('update:modelValue', raw === '' && props.allowEmpty ? '' : Number(raw));
};

const normalizeCurrent = () => {
  emit('update:modelValue', normalize(props.modelValue));
};

const stepValue = (direction) => {
  const current = Number(props.modelValue);
  if (!Number.isFinite(current)) {
    emit('update:modelValue', normalize(Number.isFinite(props.min) ? props.min : 0));
    return;
  }
  const base = current;
  emit('update:modelValue', normalize(base + direction * props.step));
};
</script>

<style scoped>
.battle-number-control {
  position: relative;
  min-width: 0;
}

.battle-number-input {
  width: 100%;
  min-height: 38px;
  appearance: textfield;
  -moz-appearance: textfield;
  padding: 8px 38px 8px 11px;
  border: 1px solid var(--border-main);
  border-radius: 7px;
  background: var(--bg-main);
  color: var(--text-main);
  font-size: var(--font-ui-sm);
  font-weight: 900;
  outline: none;
  transition: border-color .15s ease, box-shadow .15s ease;
}

.battle-number-input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 15%, transparent);
}

.battle-number-input::-webkit-inner-spin-button,
.battle-number-input::-webkit-outer-spin-button {
  margin: 0;
  -webkit-appearance: none;
}

.battle-number-steppers {
  position: absolute;
  inset-block: 4px;
  right: 4px;
  width: 27px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--border-main) 82%, transparent);
  border-radius: 5px;
  background: color-mix(in srgb, var(--bg-card) 92%, transparent);
}

.battle-number-steppers button {
  flex: 1;
  display: grid;
  place-items: center;
  border: 0;
  background: transparent;
  color: var(--text-secondary);
}

.battle-number-steppers button + button {
  border-top: 1px solid color-mix(in srgb, var(--border-main) 75%, transparent);
}

.battle-number-steppers button:hover {
  background: var(--btn-bg);
  color: white;
}

.battle-number-arrow {
  width: 0;
  height: 0;
  border-inline: 4px solid transparent;
}

.battle-number-arrow.up { border-bottom: 5px solid currentColor; }
.battle-number-arrow.down { border-top: 5px solid currentColor; }
</style>
