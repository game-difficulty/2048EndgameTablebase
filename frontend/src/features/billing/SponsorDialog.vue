<template>
  <div v-if="open" class="sponsor-overlay">
    <div class="absolute inset-0 bg-slate-950/42 backdrop-blur-sm" @click="$emit('close')" />
    <section class="sponsor-panel">
      <div class="flex items-start justify-between gap-4">
        <div>
          <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('billing.kicker') }}</div>
          <h2 class="mt-1 ui-metric font-black text-text-main">{{ $t('billing.title') }}</h2>
          <p class="mt-2 ui-body text-text-secondary">{{ $t('billing.subtitle') }}</p>
        </div>
        <button type="button" class="action-btn-small sponsor-close" @click="$emit('close')">
          {{ $t('common.close') }}
        </button>
      </div>

      <div class="mt-5 grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div class="sponsor-qr-card">
          <img class="sponsor-qr" src="/payments/wechat-sponsor-qr.png" :alt="$t('billing.qrAlt')" />
          <div class="mt-3 text-center ui-caption font-black uppercase text-text-secondary">
            {{ $t('billing.wechatQr') }}
          </div>
        </div>

        <div class="grid gap-3">
          <div v-for="tier in tiers" :key="tier.amount" class="sponsor-tier">
            <div>
              <div class="text-2xl font-black text-text-main">¥{{ tier.amount }}</div>
              <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('billing.sponsorAmount') }}</div>
            </div>
            <div class="text-right">
              <div class="text-2xl font-black text-accent">{{ formatTokens(tier.tokens) }}</div>
              <div class="ui-caption font-black uppercase text-text-secondary">token</div>
            </div>
          </div>

          <div class="sponsor-note">
            <div class="font-black text-text-main">{{ $t('billing.manualTitle') }}</div>
            <p class="sponsor-benefit">{{ $t('billing.weeklyBenefit') }}</p>
            <p>{{ $t('billing.manualNote') }}</p>
            <p>{{ $t('billing.remarkNote', { identity: paymentRemark }) }}</p>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed } from 'vue';

const props = defineProps({
  open: Boolean,
  user: {
    type: Object,
    default: null,
  },
});

defineEmits(['close']);

const tiers = [
  { amount: '9.9', tokens: 100000 },
  { amount: '99', tokens: 2000000 },
];

const paymentRemark = computed(() => (
  props.user?.email || props.user?.display_name || '-'
));

const formatTokens = (value) => Number(value || 0).toLocaleString();
</script>

<style scoped>
.sponsor-overlay {
  position: absolute;
  inset: 0;
  z-index: 116;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
}

.sponsor-panel {
  position: relative;
  z-index: 1;
  width: min(42rem, 100%);
  max-height: calc(100vh - 2rem);
  overflow-y: auto;
  border: 1px solid var(--border-main);
  border-radius: 1.5rem;
  background: color-mix(in srgb, var(--bg-card) 96%, transparent);
  padding: 1.5rem;
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.34);
}

.sponsor-close {
  min-width: 4.5rem;
}

.sponsor-qr-card,
.sponsor-tier,
.sponsor-note {
  border: 1px solid var(--border-main);
  border-radius: 1rem;
  background: color-mix(in srgb, var(--bg-main) 58%, transparent);
}

.sponsor-qr-card {
  padding: 1rem;
}

.sponsor-qr {
  display: block;
  width: min(17rem, 100%);
  margin-inline: auto;
  border-radius: 0.75rem;
  background: white;
}

.sponsor-tier {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding: 1rem;
}

.sponsor-note {
  display: grid;
  gap: 0.35rem;
  padding: 1rem;
  color: var(--text-secondary);
  font-size: var(--font-ui-sm);
  font-weight: 800;
  line-height: 1.45;
}

.sponsor-benefit {
  color: var(--accent);
}

@media (max-width: 520px) {
  .sponsor-overlay {
    padding: 0.75rem;
  }

  .sponsor-panel {
    border-radius: 1.1rem;
    padding: 1rem;
  }
}
</style>
