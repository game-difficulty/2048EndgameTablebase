<template>
  <div
    ref="root"
    class="relative h-[64px] w-full select-none"
    @contextmenu.prevent="openThresholdEditor"
  >
    <input
      :value="sliderValue"
      type="range"
      :min="0"
      :max="sliderMax"
      step="1"
      class="absolute inset-x-0 top-1/2 z-20 h-6 w-full -translate-y-1/2 cursor-pointer appearance-none bg-transparent opacity-0"
      :disabled="sliderMax <= 0"
      @input="handleInput"
    />
    <div class="replay-slider-track pointer-events-none absolute inset-x-0 top-1/2 -translate-y-1/2">
      <div class="replay-slider-progress absolute inset-y-0 left-0 rounded-full" :style="{ width: `${progressPercent}%` }" />
    </div>
    <div
      v-for="marker in markers"
      :key="marker.index"
      class="pointer-events-none absolute top-[44px] z-10 -translate-x-1/2"
      :style="{ left: `${marker.left}%` }"
    >
      <div
        class="h-0 w-0 border-x-[5px] border-t-0 border-b-[9px] border-x-transparent"
        :style="{ borderBottomColor: marker.color }"
      />
    </div>
    <div
      class="replay-slider-thumb pointer-events-none absolute top-1/2 z-10 -translate-x-1/2 -translate-y-1/2 rounded-full"
      :style="{ left: `${progressPercent}%` }"
    />
    <div
      v-if="editorOpen"
      class="absolute left-1/2 top-full z-30 mt-2 w-60 -translate-x-1/2 rounded-xl border border-border-main/80 bg-bg-card/92 p-3 shadow-[0_14px_32px_rgba(0,0,0,0.22)] backdrop-blur-sm"
    >
      <div class="ui-kicker font-black uppercase tracking-[0.18em] text-text-secondary">
        {{ t('replay.slider.title') }}
      </div>
      <div class="mt-1 ui-control font-black text-text-main">
        {{ t('replay.slider.description') }}
      </div>
      <div class="mt-2 flex items-center gap-2">
        <div class="group relative flex-1">
          <input
            ref="editorInput"
            v-model="editorValue"
            type="number"
            min="0"
            max="1"
            step="0.001"
            class="slider-number-input w-full rounded-lg border border-border-main bg-bg-main px-3 py-2 pr-9 ui-control font-black tabular-nums text-text-main outline-none transition-colors focus:border-accent"
            @keydown.enter.prevent="confirmThreshold"
            @keydown.esc.prevent="closeThresholdEditor"
          />
          <div class="pointer-events-none absolute inset-y-1 right-1 flex w-5 flex-col overflow-hidden rounded-md border border-border-main/70 bg-bg-card/90 opacity-0 shadow-sm transition-opacity duration-150 group-hover:pointer-events-auto group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:opacity-100">
            <button type="button" class="slider-spin-btn border-b border-border-main/60" :aria-label="t('replay.slider.increase')" @click="stepEditorValue(1)">
              <span class="slider-spin-icon slider-spin-icon-up" />
            </button>
            <button type="button" class="slider-spin-btn" :aria-label="t('replay.slider.decrease')" @click="stepEditorValue(-1)">
              <span class="slider-spin-icon slider-spin-icon-down" />
            </button>
          </div>
        </div>
        <button type="button" class="slider-btn slider-btn-primary" @click="confirmThreshold">{{ t('common.apply') }}</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps({
  losses: { type: Array, default: () => [] },
  currentStep: { type: Number, default: 0 },
  threshold: { type: Number, default: 1 },
});

const emit = defineEmits(['update-step', 'update-threshold']);
const { t } = useI18n();

const root = ref(null);
const editorInput = ref(null);
const editorOpen = ref(false);
const editorValue = ref('1');

const sliderMax = computed(() => Math.max((props.losses?.length || 0) - 1, 0));
const sliderValue = computed(() => Math.min(Math.max(props.currentStep || 0, 0), sliderMax.value));
const progressPercent = computed(() => {
  if (sliderMax.value <= 0) return 0;
  return (sliderValue.value / sliderMax.value) * 100;
});

const colorForLoss = (loss) => {
  if (loss >= 1 - 3e-10) return 'rgba(127, 255, 127, 0.9)';
  if (loss >= 0.999) return 'rgba(128, 255, 0, 0.8)';
  if (loss >= 0.99) return 'rgba(0, 255, 0, 0.78)';
  if (loss >= 0.975) return 'rgba(255, 245, 0, 0.82)';
  if (loss >= 0.9) return 'rgba(255, 165, 0, 0.84)';
  if (loss >= 0.75) return 'rgba(255, 0, 127, 0.82)';
  return 'rgba(255, 0, 0, 0.82)';
};

const markers = computed(() => {
  const losses = Array.isArray(props.losses) ? props.losses.map(Number).filter(Number.isFinite) : [];
  if (!losses.length) return [];
  const sorted = [...losses].sort((a, b) => a - b);
  const qIndex = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * 0.1)));
  const threshold = Math.min(sorted[qIndex], Number(props.threshold) || 1);
  return losses
    .map((loss, index) => ({ loss, index }))
    .filter(({ loss }) => loss < 1 && loss < threshold)
    .map(({ loss, index }) => ({
      index,
      left: sliderMax.value > 0 ? (index / sliderMax.value) * 100 : 0,
      color: colorForLoss(loss),
    }));
});

const handleInput = (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement)) return;
  emit('update-step', target.valueAsNumber || 0);
};

const closeThresholdEditor = () => {
  editorOpen.value = false;
};

const openThresholdEditor = async () => {
  editorValue.value = String(Number(props.threshold) || 1);
  editorOpen.value = true;
  await nextTick();
  editorInput.value?.focus();
  editorInput.value?.select();
};

const confirmThreshold = () => {
  const next = Number(editorValue.value);
  if (!Number.isFinite(next)) {
    closeThresholdEditor();
    return;
  }
  emit('update-threshold', Math.min(1, Math.max(0, next)));
  closeThresholdEditor();
};

const stepEditorValue = (direction) => {
  const current = Number(editorValue.value);
  const base = Number.isFinite(current) ? current : Number(props.threshold) || 1;
  const next = Math.min(1, Math.max(0, base + (direction * 0.001)));
  editorValue.value = next.toFixed(3).replace(/\.?0+$/, (match) => (match.startsWith('.') ? '' : match));
};

const handlePointerDown = (event) => {
  if (!editorOpen.value) return;
  if (root.value?.contains(event.target)) return;
  closeThresholdEditor();
};

const handleKeyDown = (event) => {
  if (event.key === 'Escape') closeThresholdEditor();
};

onMounted(() => {
  document.addEventListener('pointerdown', handlePointerDown);
  document.addEventListener('keydown', handleKeyDown);
});

onBeforeUnmount(() => {
  document.removeEventListener('pointerdown', handlePointerDown);
  document.removeEventListener('keydown', handleKeyDown);
});
</script>

<style scoped>
.replay-slider-track {
  height: 0.55rem;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--accent) 14%, var(--border-main));
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0.02)) padding-box,
    var(--range-track-gradient) border-box;
  box-shadow:
    inset 0 1px 1px rgba(255, 255, 255, 0.2),
    inset 0 0 0 999px var(--range-track-bg);
}

.replay-slider-progress {
  background: var(--range-track-gradient);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.16),
    0 0 10px color-mix(in srgb, var(--accent) 18%, transparent);
}

.replay-slider-thumb {
  width: 1.2rem;
  height: 1.2rem;
  border: 1px solid var(--range-thumb-border);
  background: var(--range-thumb-gradient);
  box-shadow: var(--range-thumb-shadow);
}

.slider-btn {
  border-radius: 0.65rem;
  border: 1px solid var(--border-main);
  background: color-mix(in srgb, var(--bg-main) 74%, transparent);
  color: var(--text-main);
  padding: 0.45rem 0.7rem;
  font-size: var(--font-ui-xs);
  font-weight: 900;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  transition: all 0.18s ease;
}

.slider-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
}

.slider-btn-primary {
  background: color-mix(in srgb, var(--accent) 14%, var(--bg-main));
  border-color: color-mix(in srgb, var(--accent) 42%, var(--border-main));
}

.slider-number-input {
  appearance: textfield;
  -moz-appearance: textfield;
}

.slider-number-input::-webkit-inner-spin-button,
.slider-number-input::-webkit-outer-spin-button {
  -webkit-appearance: none;
  margin: 0;
}

.slider-spin-btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  color: var(--text-secondary);
  transition: background-color 0.15s ease, color 0.15s ease;
}

.slider-spin-btn:hover {
  background: var(--btn-bg);
  color: #fff;
}

.slider-spin-icon {
  display: block;
  width: 0;
  height: 0;
  border-left: 4px solid transparent;
  border-right: 4px solid transparent;
}

.slider-spin-icon-up {
  border-bottom: 6px solid currentColor;
}

.slider-spin-icon-down {
  border-top: 6px solid currentColor;
}
</style>
