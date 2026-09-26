<template>
  <section class="token-sources">
    <h3>{{ $t(`${copyKey}.title`) }}</h3>
    <p v-for="source in ['note', 'recharge', 'lucky', 'envelope']" :key="source">
      <template v-for="(part, index) in emphasisParts($t(`${copyKey}.${source}`))" :key="index"><strong v-if="part.bold">{{ part.text }}</strong><template v-else>{{ part.text }}</template></template>
    </p>
    <p>
      <template v-for="(part, index) in emphasisParts($t(`${copyKey}.weekly`))" :key="index"><strong v-if="part.bold">{{ part.text }}</strong><template v-else>{{ part.text }}</template></template>
    </p>
    <div class="weekly-reward-table-wrap">
      <table class="weekly-reward-table">
        <thead><tr>
          <th>{{ $t('billing.quotaGuide.earn.weeklyTable.rank') }}</th>
          <th>{{ $t('billing.quotaGuide.earn.weeklyTable.main') }}</th>
          <th>{{ $t('billing.quotaGuide.earn.weeklyTable.adversarial') }}</th>
          <th>{{ $t('billing.quotaGuide.earn.weeklyTable.play4') }}</th>
          <th>{{ $t('billing.quotaGuide.earn.weeklyTable.playOther') }}</th>
        </tr></thead>
        <tbody><tr v-for="row in weeklyRewards" :key="row.rank">
          <th>{{ row.rank }}</th>
          <td>{{ formatReward(row.main) }}</td>
          <td>{{ formatReward(row.adversarial) }}</td>
          <td>{{ formatReward(row.play4) }}</td>
          <td>{{ formatReward(row.playOther) }}</td>
        </tr></tbody>
      </table>
    </div>
    <p><template v-for="(part, index) in emphasisParts($t(`${copyKey}.trophy`))" :key="index"><strong v-if="part.bold">{{ part.text }}</strong><template v-else>{{ part.text }}</template></template></p>
  </section>
</template>

<script setup>
import { useI18n } from 'vue-i18n';
import { emphasisParts } from './tokenSourceText.js';
defineProps({ copyKey: { type: String, default: 'billing.quotaGuide.earn' } });
const { locale } = useI18n();
const main = [10000, 8000, 5000, 3000, 2000];
const adversarial = [5000, 4000, 3000, 2000, 1000];
const play4 = [36000, 24000, 16000, 12000, 10000, 8000, 6000, 5000, 4000, 3000];
const playOther = [16000, 12000, 10000, 8000, 6000, 5000, 4000, 3000, 2000, 1000];
const weeklyRewards = Array.from({ length: 10 }, (_, index) => ({ rank: index + 1,
  main: main[index], adversarial: adversarial[index], play4: play4[index], playOther: playOther[index] }));
const formatReward = value => value == null ? '—' : new Intl.NumberFormat(locale.value).format(value);
</script>

<style scoped>
.token-sources { display: grid; grid-template-columns: minmax(0, 1fr); gap: 0.6rem; color: var(--text-secondary); font-size: var(--font-ui-sm); line-height: 1.6; }
h3 { color: var(--text-main); font-size: var(--font-ui-md); font-weight: 950; }
strong { color: var(--text-main); font-weight: 850; }
.weekly-reward-table-wrap { max-width: 100%; overflow-x: auto; border: 1px solid var(--border-main); border-radius: .65rem; }
.weekly-reward-table { width: 100%; min-width: 42rem; border-collapse: collapse; color: var(--text-main); font-size: var(--font-ui-xs); font-variant-numeric: tabular-nums; }
.weekly-reward-table th, .weekly-reward-table td { padding: .5rem .65rem; border-right: 1px solid var(--border-main); border-bottom: 1px solid var(--border-main); text-align: right; white-space: nowrap; }
.weekly-reward-table th:first-child { text-align: center; }
.weekly-reward-table thead th { background: color-mix(in srgb, var(--bg-main) 76%, transparent); color: var(--text-secondary); font-weight: 900; }
.weekly-reward-table tbody tr:last-child > * { border-bottom: 0; }
.weekly-reward-table tr > *:last-child { border-right: 0; }
</style>
