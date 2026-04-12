<template>
  <div
    ref="shellRef"
    class="countdown-shell"
    :class="{ 'is-counting': active, 'is-disabled': disabled }"
  >
    <span v-if="active && width > 0 && height > 0" class="countdown-ring-shell" aria-hidden="true">
      <svg class="countdown-ring" :viewBox="`0 0 ${width} ${height}`" preserveAspectRatio="none">
        <rect
          class="countdown-ring-progress"
          :x="strokeHalf"
          :y="strokeHalf"
          :width="ringWidth"
          :height="ringHeight"
          :rx="ringRadius"
          :ry="ringRadius"
          pathLength="100"
          :transform="progressTransform"
          :style="{ strokeDasharray: `${clampedProgress * 100} 100` }"
        />
      </svg>
    </span>
    <button
      type="button"
      class="countdown-btn"
      :disabled="disabled"
      @click="$emit('click')"
    >
      <span class="countdown-label"><slot /></span>
      <span
        v-if="countdownLabel"
        class="countdown-meta"
      >
        {{ countdownLabel }}
      </span>
    </button>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue';

const props = defineProps({
  active: { type: Boolean, default: false },
  progress: { type: Number, default: 0 },
  disabled: { type: Boolean, default: false },
  countdownLabel: { type: String, default: '' },
});

defineEmits(['click']);

const shellRef = ref(null);
const width = ref(0);
const height = ref(0);
const radius = ref(12);
const strokeWidth = 2.6;

let resizeObserver = null;

const strokeHalf = computed(() => strokeWidth / 2);
const clampedProgress = computed(() => Math.max(0, Math.min(1, Number(props.progress) || 0)));
const ringWidth = computed(() => Math.max(0, width.value - strokeWidth));
const ringHeight = computed(() => Math.max(0, height.value - strokeWidth));
const ringRadius = computed(() => Math.max(0, radius.value - strokeHalf.value));
const progressTransform = computed(() => `translate(${width.value} 0) scale(-1 1)`);

const syncMetrics = () => {
  const el = shellRef.value;
  if (!el) return;
  const rect = el.getBoundingClientRect();
  const styles = window.getComputedStyle(el);
  width.value = Math.max(0, Math.round(rect.width));
  height.value = Math.max(0, Math.round(rect.height));
  radius.value = Number.parseFloat(styles.borderTopLeftRadius) || 12;
};

onMounted(() => {
  syncMetrics();
  if ('ResizeObserver' in window) {
    resizeObserver = new ResizeObserver(() => {
      syncMetrics();
    });
    if (shellRef.value) {
      resizeObserver.observe(shellRef.value);
    }
  } else {
    window.addEventListener('resize', syncMetrics);
  }
});

onUnmounted(() => {
  resizeObserver?.disconnect();
  if (!resizeObserver) {
    window.removeEventListener('resize', syncMetrics);
  }
});
</script>

<style scoped>
.countdown-shell {
  position: relative;
  width: 100%;
  border-radius: 0.75rem;
  isolation: isolate;
}

.countdown-btn {
  position: relative;
  z-index: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  border-radius: inherit;
  border: 1px solid var(--border-main);
  background: var(--bg-card);
  color: var(--text-main);
  padding: 0.65rem 0.9rem;
  font-size: var(--font-ui-sm);
  font-weight: 900;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  transition: all 0.2s ease;
  appearance: none;
}

.countdown-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
}

.countdown-shell.is-disabled .countdown-btn {
  cursor: not-allowed;
  opacity: 0.45;
}

.countdown-shell.is-counting .countdown-btn {
  border-color: rgba(245, 158, 11, 0.88);
  background: color-mix(in srgb, var(--bg-card) 82%, rgba(245, 158, 11, 0.18));
  box-shadow:
    inset 0 0 0 1px rgba(251, 191, 36, 0.18),
    0 0 0 1px rgba(245, 158, 11, 0.26),
    0 10px 24px rgba(245, 158, 11, 0.14);
}

.countdown-ring-shell {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 0;
}

.countdown-ring {
  display: block;
  width: 100%;
  height: 100%;
  overflow: visible;
}

.countdown-ring-progress {
  fill: none;
  vector-effect: non-scaling-stroke;
}

.countdown-ring-progress {
  stroke: rgba(245, 158, 11, 1);
  stroke-width: 5.4;
  stroke-linecap: round;
  transition: stroke-dasharray 0.05s linear;
  filter: drop-shadow(0 0 6px rgba(245, 158, 11, 0.55));
}

.countdown-label,
.countdown-meta {
  position: relative;
  z-index: 1;
}

.countdown-meta {
  margin-left: 0.5rem;
  font-size: var(--font-ui-2xs);
  font-weight: 900;
  font-variant-numeric: tabular-nums;
  opacity: 0.88;
}
</style>
