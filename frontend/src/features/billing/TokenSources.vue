<template>
  <section class="token-sources">
    <h3>{{ $t(`${copyKey}.title`) }}</h3>
    <p>{{ $t(`${copyKey}.note`) }}</p>
    <p v-for="source in ['recharge', 'lucky', 'envelope', 'weekly', 'trophy']" :key="source">
      {{ $t(`${copyKey}.${source}`) }}
    </p>
  </section>
</template>

<script setup>
defineProps({ copyKey: { type: String, default: 'billing.quotaGuide.earn' } });
</script>

<style scoped>
.token-sources { display: grid; grid-template-columns: minmax(0, 1fr); gap: 0.6rem; color: var(--text-secondary); font-size: var(--font-ui-sm); line-height: 1.6; }
h3 { color: var(--text-main); font-size: var(--font-ui-md); font-weight: 950; }
</style>
