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
      <div v-else-if="panel.type === 'countdown'" class="mt-2 rounded-xl bg-bg-main/60 px-3 py-3 text-center">
        <div class="font-black text-prominent ui-text-xl tabular-nums">{{ countdownText(panel) }}</div>
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
import { computed, onMounted, onUnmounted, ref } from 'vue';
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
let timer = null;

const hudPanels = computed(() => props.hud?.customPanels || []);
const visiblePanels = computed(() =>
  hudPanels.value.filter((panel) => panel?.type !== 'patternText' && panel?.type !== 'targetPattern')
);

const emitCustomAction = (key, phase) => {
  if (!key) return;
  emit('custom-action', { key, phase });
};

const countdownText = (panel) => {
  const base = Number(panel.remainingMs || 0);
  const syncedAt = Number(panel.syncedAt || now.value);
  const elapsed = panel.running ? Math.max(0, now.value - syncedAt) : 0;
  const remaining = Math.max(0, base - elapsed);
  const minutes = Math.floor(remaining / 60000);
  const seconds = Math.floor((remaining % 60000) / 1000);
  const hundredths = Math.floor((remaining % 1000) / 10);
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(hundredths).padStart(2, '0')}`;
};

onMounted(() => {
  timer = window.setInterval(() => {
    now.value = Date.now();
  }, 80);
});

onUnmounted(() => {
  if (timer) window.clearInterval(timer);
});
</script>

<style scoped>
.minigame-pre {
  margin: 0;
  font-family: Consolas, "Courier New", monospace;
  font-size: var(--font-ui-sm);
  line-height: 1.35;
  white-space: pre-wrap;
  color: var(--text-main);
}
</style>
