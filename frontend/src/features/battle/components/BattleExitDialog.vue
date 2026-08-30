<template>
  <div class="battle-exit-backdrop" @click.self="$emit('cancel')">
    <section class="battle-exit-dialog" role="alertdialog" aria-modal="true" aria-labelledby="battle-exit-title">
      <span class="ui-caption font-black uppercase text-text-secondary">{{ $t('battle.forfeit.kicker') }}</span>
      <h2 id="battle-exit-title">{{ $t('battle.forfeit.title') }}</h2>
      <p>{{ $t('battle.forfeit.body') }}</p>
      <div class="battle-exit-actions">
        <button type="button" :disabled="pending" @click="$emit('cancel')">{{ $t('battle.actions.cancel') }}</button>
        <button type="button" class="danger" :disabled="pending" @click="$emit('confirm')">
          {{ pending ? $t('battle.actions.processing') : $t('battle.forfeit.confirm') }}
        </button>
      </div>
    </section>
  </div>
</template>

<script setup>
defineProps({ pending: { type: Boolean, default: false } });
defineEmits(['cancel', 'confirm']);
</script>

<style scoped>
.battle-exit-backdrop { position: absolute; inset: 0; z-index: 230; display: grid; place-items: center; padding: 28px; background: rgba(8,14,28,.62); backdrop-filter: blur(5px); }
.battle-exit-dialog { width: min(440px, 94%); padding: 24px; border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-card); box-shadow: 0 28px 80px rgba(0,0,0,.32); }
.battle-exit-dialog h2 { margin: 6px 0 8px; color: var(--text-main); font-size: 24px; font-weight: 900; }
.battle-exit-dialog p { margin: 0; color: var(--text-secondary); font-size: var(--font-ui-sm); line-height: 1.65; }
.battle-exit-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; margin-top: 20px; }
.battle-exit-actions button { min-height: 42px; border: 1px solid var(--border-main); border-radius: 7px; background: var(--bg-card); color: var(--text-main); font-weight: 900; }
.battle-exit-actions button.danger { border-color: #d94f56; background: #d94f56; color: white; }
.battle-exit-actions button:disabled { cursor: wait; opacity: .55; }
</style>
