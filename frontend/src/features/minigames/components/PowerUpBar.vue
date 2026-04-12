<template>
  <div class="powerup-bar-shell w-full rounded-md p-4 flex flex-col space-y-3 shadow-sm">
    <div class="flex items-center justify-between gap-3">
      <span
        class="powerup-header-text"
        :class="{ 'is-hint': interaction?.active && hintText }"
      >
        {{ interaction?.active && hintText ? hintText : $t('minigames.play.powerUps') }}
      </span>
      <button
        v-if="interaction?.active"
        type="button"
        class="powerup-cancel-btn"
        :title="$t('minigames.play.cancel')"
        :aria-label="$t('minigames.play.cancel')"
        @click="$emit('cancel')"
      >
        ×
      </button>
    </div>
    <div class="grid grid-cols-3 gap-2">
      <button
        v-for="entry in entries"
        :key="entry.key"
        type="button"
        :disabled="!isEntryEnabled(entry.key)"
        :class="[
          'action-btn-small',
          powerups?.activeMode === entry.key ? 'btn-prominent' : ''
        ]"
        @click="$emit('use', entry.key)"
      >
        <span class="mr-2">{{ entry.label }}</span>
        <span class="pill-badge">{{ entry.count }}</span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps({
  powerups: {
    type: Object,
    default: () => ({
      enabled: false,
      counts: { bomb: 0, glove: 0, twist: 0 },
      activeMode: null,
    }),
  },
  interaction: {
    type: Object,
    default: () => ({
      active: false,
      mode: null,
      hintKey: '',
    }),
  },
});

defineEmits(['use', 'cancel']);

const { t } = useI18n();

const entries = computed(() => [
  { key: 'bomb', label: t('minigames.powerups.bomb'), count: props.powerups?.counts?.bomb ?? 0 },
  { key: 'glove', label: t('minigames.powerups.glove'), count: props.powerups?.counts?.glove ?? 0 },
  { key: 'twist', label: t('minigames.powerups.twist'), count: props.powerups?.counts?.twist ?? 0 },
]);

const hintText = computed(() => {
  const hintKey = props.interaction?.hintKey;
  return hintKey ? t(hintKey) : '';
});

const isEntryEnabled = (key) => {
  if (!props.powerups?.enabled) return false;
  if ((props.powerups?.counts?.[key] ?? 0) <= 0) return false;
  return !props.interaction?.active || props.powerups?.activeMode === key;
};
</script>

<style scoped>
.powerup-bar-shell {
  background:
    linear-gradient(135deg, rgba(255, 225, 150, 0.22) 0%, rgba(255, 205, 120, 0.12) 38%, rgba(255, 255, 255, 0) 100%),
    var(--ctrl-bg);
  border: 1px solid color-mix(in srgb, var(--accent) 18%, var(--border-main));
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.32),
    0 4px 14px rgba(245, 158, 11, 0.08);
}

[data-theme='dark'] .powerup-bar-shell {
  background:
    linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(37, 99, 235, 0.12) 38%, rgba(15, 23, 42, 0) 100%),
    var(--ctrl-bg);
  border-color: color-mix(in srgb, #38bdf8 26%, var(--border-main));
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.06),
    0 6px 18px rgba(14, 165, 233, 0.12);
}

.powerup-header-text {
  min-height: 1.9rem;
  display: inline-flex;
  align-items: center;
  color: var(--text-main);
  font-size: var(--font-ui-sm);
  font-weight: 700;
  opacity: 0.8;
  line-height: 1.2;
}

.powerup-header-text.is-hint {
  opacity: 1;
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 900;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.powerup-cancel-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.9rem;
  min-width: 1.9rem;
  height: 1.9rem;
  border-radius: 999px;
  border: 1px solid var(--border-main);
  background: color-mix(in srgb, var(--bg-main) 72%, transparent);
  color: var(--text-secondary);
  font-size: 1rem;
  font-weight: 900;
  line-height: 1;
  transition: all 0.18s ease;
}

.powerup-cancel-btn:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: color-mix(in srgb, var(--accent) 10%, var(--bg-main));
}
</style>
