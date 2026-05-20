<template>
  <div class="w-full flex flex-col gap-3">
    <div
      v-for="panel in visiblePanels"
      :key="panel.title + panel.type"
      class="console-card"
    >
      <div class="console-card-header">
        <span>{{ panel.title }}</span>
      </div>
      <div v-if="panel.type === 'patternText'" class="mt-2 rounded-lg bg-bg-main/60 px-3 py-2">
        <pre class="minigame-pre">{{ (panel.lines || []).join('\n') }}</pre>
      </div>
      <div
        v-else-if="panel.type === 'countdown'"
        class="countdown-panel mt-2 rounded-xl bg-bg-main/60 px-3 py-3 text-center"
        :class="{ 'countdown-panel-bonus': bonusActive }"
      >
        <div class="font-black text-prominent ui-text-xl tabular-nums">{{ countdownText(panel) }}</div>
        <Transition name="countdown-bonus">
          <div v-if="bonusText" class="countdown-bonus-badge tabular-nums">{{ bonusText }}</div>
        </Transition>
      </div>
      <div v-else-if="panel.type === 'actionButton'" class="mt-2 rounded-lg bg-bg-main/60 px-3 py-3">
        <button
          type="button"
          class="action-btn w-full"
          :class="panel.pressed ? 'btn-prominent' : ''"
          :disabled="!panel.enabled"
          @pointerdown.prevent="emitCustomAction(panel.key, 'start')"
          @pointerup.prevent="emitCustomAction(panel.key, 'end')"
          @pointerleave="emitCustomAction(panel.key, 'end')"
          @pointercancel="emitCustomAction(panel.key, 'cancel')"
          @click.prevent="!panel.hold && emitCustomAction(panel.key, 'trigger')"
        >
          <span>{{ panel.label }}</span>
          <span v-if="panel.meta" class="pill-badge">{{ panel.meta }}</span>
        </button>
      </div>
      <div v-else class="mt-2 rounded-lg bg-bg-main/60 px-3 py-2">
        <div class="ui-body font-black text-text-main">
          {{ panel.value }} <span v-if="panel.suffix" class="ui-caption uppercase text-text-secondary">{{ panel.suffix }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps({
  hud: {
    type: Object,
    default: () => ({ customPanels: [] }),
  },
});

const emit = defineEmits(['custom-action']);

const { t } = useI18n();
const now = ref(Date.now());
const bonusText = ref('');
const bonusActive = ref(false);
const lastCountdownRemaining = ref(null);
let timer = null;
let bonusTimer = null;

const hudPanels = computed(() => props.hud?.customPanels || []);
const visiblePanels = computed(() =>
  hudPanels.value.filter((panel) => panel?.type !== 'patternText' && panel?.type !== 'targetPattern')
);
const countdownPanel = computed(() => hudPanels.value.find((panel) => panel?.type === 'countdown') || null);

const emitCustomAction = (key, phase) => {
  if (!key) return;
  emit('custom-action', { key, phase });
};

const remainingMsForPanel = (panel, nowMs = now.value) => {
  const base = Number(panel.remainingMs || 0);
  const syncedAt = Number(panel.syncedAt || nowMs);
  const elapsed = panel.running ? Math.max(0, nowMs - syncedAt) : 0;
  return Math.max(0, base - elapsed);
};

const formatTimerText = (remaining) => {
  const minutes = Math.floor(remaining / 60000);
  const seconds = Math.floor((remaining % 60000) / 1000);
  const hundredths = Math.floor((remaining % 1000) / 10);
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(hundredths).padStart(2, '0')}`;
};

const countdownText = (panel) => formatTimerText(remainingMsForPanel(panel));

const formatBonusText = (deltaMs) => {
  const totalSeconds = Math.max(1, Math.round(Number(deltaMs || 0) / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes > 0) return `+${minutes}:${String(seconds).padStart(2, '0')}`;
  return `+${seconds}s`;
};

const clearBonusTimer = () => {
  if (bonusTimer) {
    window.clearTimeout(bonusTimer);
    bonusTimer = null;
  }
};

const showBonus = (deltaMs) => {
  clearBonusTimer();
  bonusText.value = formatBonusText(deltaMs);
  bonusActive.value = false;
  window.requestAnimationFrame(() => {
    bonusActive.value = true;
  });
  bonusTimer = window.setTimeout(() => {
    bonusText.value = '';
    bonusActive.value = false;
    bonusTimer = null;
  }, 1050);
};

watch(
  countdownPanel,
  (panel) => {
    if (!panel) {
      lastCountdownRemaining.value = null;
      bonusText.value = '';
      bonusActive.value = false;
      clearBonusTimer();
      return;
    }
    const remaining = remainingMsForPanel(panel, Date.now());
    const previous = lastCountdownRemaining.value;
    const bonusMs = Number(panel.bonusMs || 0);
    if (previous !== null && bonusMs > 0) {
      showBonus(bonusMs);
    } else if (previous !== null && panel.running) {
      const delta = remaining - previous;
      if (delta > 1000) {
        showBonus(delta);
      }
    }
    lastCountdownRemaining.value = remaining;
  },
  { immediate: true }
);

onMounted(() => {
  timer = window.setInterval(() => {
    now.value = Date.now();
  }, 80);
});

onUnmounted(() => {
  if (timer) window.clearInterval(timer);
  clearBonusTimer();
});
</script>

<style scoped>
.countdown-panel {
  position: relative;
  overflow: hidden;
}

.countdown-panel-bonus {
  animation: countdown-panel-pulse 420ms ease-out;
}

.countdown-bonus-badge {
  position: absolute;
  right: 0.75rem;
  top: 0.45rem;
  border-radius: 999px;
  padding: 0.16rem 0.45rem;
  background: color-mix(in srgb, var(--text-prominent) 20%, transparent);
  color: var(--text-prominent);
  font-size: var(--font-ui-xs);
  font-weight: 900;
  line-height: 1;
  pointer-events: none;
  text-shadow: 0 1px 0 rgba(255, 255, 255, 0.32);
}

.countdown-bonus-enter-active {
  animation: countdown-bonus-float 900ms ease-out forwards;
}

.countdown-bonus-leave-active {
  transition: opacity 140ms ease;
}

.countdown-bonus-leave-to {
  opacity: 0;
}

.minigame-pre {
  margin: 0;
  font-family: Consolas, "Courier New", monospace;
  font-size: var(--font-ui-sm);
  line-height: 1.35;
  white-space: pre-wrap;
  color: var(--text-main);
}

@keyframes countdown-panel-pulse {
  0% {
    box-shadow: inset 0 0 0 0 color-mix(in srgb, var(--text-prominent) 0%, transparent);
  }
  42% {
    box-shadow: inset 0 0 0 2px color-mix(in srgb, var(--text-prominent) 36%, transparent);
  }
  100% {
    box-shadow: inset 0 0 0 0 color-mix(in srgb, var(--text-prominent) 0%, transparent);
  }
}

@keyframes countdown-bonus-float {
  0% {
    opacity: 0;
    transform: translateY(0.35rem) scale(0.94);
  }
  18% {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
  100% {
    opacity: 0;
    transform: translateY(-0.9rem) scale(1.04);
  }
}
</style>
