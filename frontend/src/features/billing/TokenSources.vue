<template>
  <section class="token-sources">
    <h3>{{ $t(`${copyKey}.title`) }}</h3>
    <p v-for="source in ['note', 'recharge', 'lucky', 'envelope', 'weekly', 'trophy']" :key="source">
      <template v-for="(part, index) in emphasisParts($t(`${copyKey}.${source}`))" :key="index"><strong v-if="part.bold">{{ part.text }}</strong><template v-else>{{ part.text }}</template></template>
    </p>
  </section>
</template>

<script setup>
import { emphasisParts } from './tokenSourceText.js';
defineProps({ copyKey: { type: String, default: 'billing.quotaGuide.earn' } });
</script>

<style scoped>
.token-sources { display: grid; grid-template-columns: minmax(0, 1fr); gap: 0.6rem; color: var(--text-secondary); font-size: var(--font-ui-sm); line-height: 1.6; }
h3 { color: var(--text-main); font-size: var(--font-ui-md); font-weight: 950; }
strong { color: var(--text-main); font-weight: 850; }
</style>
