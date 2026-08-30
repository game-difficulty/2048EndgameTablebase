<template>
  <Teleport to="body">
    <div class="battle-rules-backdrop" role="presentation" @mousedown.self="$emit('close')">
      <section
        ref="dialogRef"
        class="battle-rules-dialog"
        role="dialog"
        aria-modal="true"
        :aria-labelledby="titleId"
        tabindex="-1"
      >
        <header class="battle-rules-header">
          <div>
            <span class="ui-caption font-black uppercase text-text-secondary">{{ $t('battle.rules.kicker') }}</span>
            <h2 :id="titleId">{{ $t('battle.rules.title') }}</h2>
            <p>{{ $t('battle.rules.subtitle') }}</p>
          </div>
          <button type="button" :aria-label="$t('common.close')" @click="$emit('close')">×</button>
        </header>

        <div class="battle-rules-content">
          <article v-for="mode in modes" :key="mode.key" class="battle-mode-rules">
            <div class="battle-mode-rule-heading">
              <span aria-hidden="true">VS</span>
              <div>
                <h3>{{ $t(mode.labelKey) }}</h3>
                <p>{{ $t(mode.summaryKey) }}</p>
              </div>
            </div>
            <ol>
              <li v-for="ruleKey in mode.ruleKeys || []" :key="ruleKey">{{ $t(ruleKey) }}</li>
            </ol>
          </article>
        </div>

        <footer>
          <button type="button" @click="$emit('close')">{{ $t('battle.rules.close') }}</button>
        </footer>
      </section>
    </div>
  </Teleport>
</template>

<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref } from 'vue';

defineProps({
  modes: { type: Array, default: () => [] },
});
const emit = defineEmits(['close']);
const dialogRef = ref(null);
const titleId = `battle-rules-${Math.random().toString(36).slice(2, 9)}`;
const handleKeydown = (event) => {
  if (event.key === 'Escape') emit('close');
};

onMounted(() => {
  window.addEventListener('keydown', handleKeydown);
  nextTick(() => dialogRef.value?.focus());
});
onBeforeUnmount(() => window.removeEventListener('keydown', handleKeydown));
</script>

<style scoped>
.battle-rules-backdrop { position: fixed; inset: 0; z-index: 420; display: grid; place-items: center; padding: 24px; background: rgba(8, 13, 24, 0.58); backdrop-filter: blur(5px); }
.battle-rules-dialog { width: min(680px, calc(100vw - 48px)); max-height: min(760px, calc(100vh - 48px)); overflow: auto; border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-card); color: var(--text-main); box-shadow: 0 30px 80px rgba(0, 0, 0, 0.3); outline: none; }
.battle-rules-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 22px; padding: 24px 26px 20px; border-bottom: 1px solid var(--border-main); }
.battle-rules-header h2 { margin: 5px 0 0; font-size: 26px; font-weight: 900; letter-spacing: 0; }
.battle-rules-header p { max-width: 520px; margin: 7px 0 0; color: var(--text-secondary); font-size: var(--font-ui-sm); line-height: 1.55; }
.battle-rules-header > button { width: 32px; height: 32px; flex: 0 0 auto; padding: 0; border: 1px solid var(--border-main); border-radius: 50%; background: var(--bg-main); color: var(--text-main); font-size: 20px; line-height: 1; }
.battle-rules-content { padding: 20px 26px; }
.battle-mode-rules + .battle-mode-rules { margin-top: 22px; padding-top: 22px; border-top: 1px solid var(--border-main); }
.battle-mode-rule-heading { display: flex; align-items: center; gap: 13px; }
.battle-mode-rule-heading > span { width: 42px; height: 42px; display: grid; flex: 0 0 auto; place-items: center; border: 2px solid var(--accent); border-radius: 50%; color: var(--accent); font: 900 11px/1 var(--font-mono, monospace); }
.battle-mode-rule-heading h3 { margin: 0; font-size: 19px; font-weight: 900; }
.battle-mode-rule-heading p { margin: 4px 0 0; color: var(--text-secondary); font-size: var(--font-ui-sm); line-height: 1.5; }
.battle-mode-rules ol { margin: 17px 0 0 55px; padding: 0 0 0 20px; color: var(--text-main); }
.battle-mode-rules li { padding-left: 5px; font-size: var(--font-ui-sm); line-height: 1.65; }
.battle-mode-rules li + li { margin-top: 8px; }
.battle-mode-rules li::marker { color: var(--accent); font-weight: 900; }
.battle-rules-dialog footer { display: flex; justify-content: flex-end; padding: 16px 26px 22px; border-top: 1px solid var(--border-main); }
.battle-rules-dialog footer button { min-width: 112px; min-height: 38px; padding: 8px 16px; border: 1px solid var(--btn-bg); border-radius: 7px; background: var(--btn-bg); color: white; font-weight: 900; }
</style>
