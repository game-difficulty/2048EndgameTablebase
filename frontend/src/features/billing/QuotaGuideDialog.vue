<template>
  <div v-if="open" class="quota-guide-overlay">
    <div class="absolute inset-0 bg-slate-950/42 backdrop-blur-sm" @click="$emit('close')" />
    <section
      class="quota-guide-panel"
      role="dialog"
      aria-modal="true"
      :aria-label="$t('billing.quotaGuide.title')"
    >
      <header class="flex items-start justify-between gap-5">
        <div>
          <div class="ui-caption font-black uppercase text-text-secondary">
            {{ $t('billing.quotaGuide.kicker') }}
          </div>
          <h2 class="mt-1 ui-metric font-black text-text-main">
            {{ $t('billing.quotaGuide.title') }}
          </h2>
          <p class="mt-2 max-w-3xl ui-body text-text-secondary">
            {{ $t('billing.quotaGuide.subtitle') }}
          </p>
        </div>
        <button type="button" class="action-btn-small quota-guide-close" @click="$emit('close')">
          {{ $t('common.close') }}
        </button>
      </header>

      <div v-if="loading" class="quota-guide-message">
        {{ $t('billing.quotaGuide.loading') }}
      </div>
      <div v-else-if="loadError" class="quota-guide-message error">
        <span>{{ $t('billing.quotaGuide.loadFailed') }}</span>
        <button type="button" class="action-btn-small" @click="loadRules">
          {{ $t('common.retry') }}
        </button>
      </div>

      <template v-else-if="rules">
        <section class="quota-guide-section">
          <div class="section-heading">
            <h3>{{ $t('billing.quotaGuide.weekly.title') }}</h3>
            <p>
              {{ $t('billing.quotaGuide.weekly.note', { days: rules.weekly_grants.interval_days }) }}
            </p>
          </div>
          <div class="weekly-tier-grid">
            <article v-for="tier in weeklyTiers" :key="tier.key" class="weekly-tier">
              <div>
                <div class="ui-control font-black text-text-main">
                  {{ $t(`billing.quotaGuide.weekly.tiers.${tier.key}.label`) }}
                </div>
                <div class="mt-1 ui-caption font-bold text-text-secondary">
                  {{ $t(`billing.quotaGuide.weekly.tiers.${tier.key}.detail`) }}
                </div>
              </div>
              <div class="weekly-amount">{{ formatTokens(tier.tokens) }}</div>
            </article>
          </div>
        </section>

        <TokenSources class="quota-guide-section quota-earn" />

        <section class="quota-guide-section">
          <div class="section-heading">
            <h3>{{ $t('billing.quotaGuide.costs.title') }}</h3>
            <p>{{ $t('billing.quotaGuide.costs.note') }}</p>
          </div>
          <div class="quota-table-wrap">
            <table class="quota-table">
              <thead>
                <tr>
                  <th>{{ $t('billing.quotaGuide.costs.tablebase') }}</th>
                  <th>{{ $t('billing.quotaGuide.costs.multiplier') }}</th>
                  <th>{{ $t('billing.quotaGuide.costs.lookupHit') }}</th>
                  <th>{{ $t('billing.quotaGuide.costs.lookupMiss') }}</th>
                  <th>{{ $t('billing.quotaGuide.costs.analysis') }}</th>
                  <th>{{ $t('billing.quotaGuide.costs.battleRoom') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="group in costRows" :key="group.patterns.join('|')">
                  <td class="patterns">{{ group.patterns.join(' / ') }}</td>
                  <td>{{ formatMultiplier(group.multiplier) }}</td>
                  <td>{{ formatTokens(group.lookupHit) }}</td>
                  <td>{{ formatTokens(group.lookupMiss) }}</td>
                  <td>{{ formatTokens(group.analysis) }}</td>
                  <td>{{ formatTokens(group.battleRoom) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div class="replay-cost-row">
            <span>{{ $t('billing.quotaGuide.costs.replay') }}</span>
            <strong>{{ formatTokens(rules.operation_costs.replay_load) }} token</strong>
          </div>
        </section>

        <section class="quota-guide-section">
          <div class="section-heading">
            <h3>{{ $t('billing.quotaGuide.thresholds.title') }}</h3>
            <p>{{ $t('billing.quotaGuide.thresholds.note') }}</p>
          </div>
          <div class="quota-table-wrap">
            <table class="quota-table threshold-table">
              <thead>
                <tr>
                  <th>{{ $t('billing.quotaGuide.thresholds.tablebase') }}</th>
                  <th>{{ $t('billing.quotaGuide.thresholds.threshold') }}</th>
                  <th>{{ $t('billing.quotaGuide.thresholds.mode') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in thresholdRows" :key="row.full_pattern">
                  <td class="patterns">{{ row.full_pattern }}</td>
                  <td>{{ formatThreshold(row.threshold) }}</td>
                  <td>{{ formatThresholdMode(row.mode) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="quota-guide-section quota-notes">
          <h3>{{ $t('billing.quotaGuide.rules.title') }}</h3>
          <p>{{ $t('billing.quotaGuide.rules.balanceOrder') }}</p>
          <p>{{ $t('billing.quotaGuide.rules.analysisCharge') }}</p>
          <p>{{ $t('billing.quotaGuide.rules.battleCharge') }}</p>
          <p>{{ $t('billing.quotaGuide.rules.replayCharge') }}</p>
          <p>{{ $t('billing.quotaGuide.rules.offline') }}</p>
        </section>
      </template>
    </section>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { getQuotaRules } from '../../services/quota/quotaClient';
import TokenSources from './TokenSources.vue';

const props = defineProps({
  open: Boolean,
});

defineEmits(['close']);

const { locale, t } = useI18n();
const rules = ref(null);
const loading = ref(false);
const loadError = ref(false);

const weeklyTiers = computed(() => {
  if (!rules.value) return [];
  return [
    { key: 'public', tokens: rules.value.weekly_grants.public },
    { key: 'invited', tokens: rules.value.weekly_grants.invited },
    { key: 'supporter', tokens: rules.value.weekly_grants.supporter },
  ];
});

const costRows = computed(() => {
  if (!rules.value) return [];
  const costs = rules.value.operation_costs;
  return rules.value.table_groups.map((group) => ({
    ...group,
    lookupHit: costs.trainer_lookup_hit * group.multiplier,
    lookupMiss: costs.trainer_lookup_miss * group.multiplier,
    analysis: costs.analysis_per_replay * group.multiplier,
    battleRoom: costs.battle_route_generation * group.multiplier,
  }));
});

const thresholdRows = computed(() => (
  Array.isArray(rules.value?.tablebase_thresholds)
    ? rules.value.tablebase_thresholds
    : []
));

const formatTokens = (value) => new Intl.NumberFormat(locale.value, {
  maximumFractionDigits: 1,
}).format(Number(value || 0));

const formatMultiplier = (value) => `${formatTokens(value)}x`;

const formatThreshold = (value) => {
  if (value === null || value === undefined) {
    return t('billing.quotaGuide.thresholds.notRecorded');
  }
  const percentage = Number(value) * 100;
  return `${new Intl.NumberFormat(locale.value, { maximumFractionDigits: 2 }).format(percentage)}%`;
};

const formatThresholdMode = (mode) => {
  if (mode === 'absolute' || mode === 'relative') {
    return t(`billing.quotaGuide.thresholds.modes.${mode}`);
  }
  return t('billing.quotaGuide.thresholds.notRecorded');
};

const loadRules = async () => {
  loading.value = true;
  loadError.value = false;
  try {
    rules.value = await getQuotaRules();
  } catch (error) {
    console.error('Failed to load quota rules', error);
    loadError.value = true;
  } finally {
    loading.value = false;
  }
};

watch(
  () => props.open,
  (open) => {
    if (open && !rules.value && !loading.value) {
      loadRules();
    }
  },
  { immediate: true },
);
</script>

<style scoped>
.quota-guide-overlay {
  position: absolute;
  inset: 0;
  z-index: 320;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
}

.quota-guide-panel {
  position: relative;
  z-index: 1;
  width: min(58rem, 100%);
  max-height: 46rem;
  overflow-y: auto;
  border: 1px solid var(--border-main);
  border-radius: 0.75rem;
  background: color-mix(in srgb, var(--bg-card) 97%, transparent);
  padding: 1.5rem;
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.34);
}

.quota-guide-close {
  flex: 0 0 auto;
  width: 5rem;
  min-width: 4.5rem;
  align-self: flex-start;
}

.quota-guide-message {
  min-height: 10rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  color: var(--text-secondary);
  font-weight: 800;
}

.quota-guide-message.error {
  color: var(--danger, #dc2626);
}

.quota-guide-section {
  margin-top: 1.25rem;
  padding-top: 1.1rem;
  border-top: 1px solid var(--border-main);
}

.section-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1.25rem;
  margin-bottom: 0.75rem;
}

.section-heading h3,
.quota-notes h3 {
  color: var(--text-main);
  font-size: var(--font-ui-md);
  font-weight: 950;
}

.section-heading p {
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 750;
  text-align: right;
}

.weekly-tier-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.75rem;
}

.weekly-tier {
  min-height: 5rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  border: 1px solid var(--border-main);
  border-radius: 0.5rem;
  background: color-mix(in srgb, var(--bg-main) 56%, transparent);
  padding: 0.8rem;
}

.weekly-amount {
  color: var(--accent);
  font-size: 1.35rem;
  font-weight: 950;
  white-space: nowrap;
}

.quota-table-wrap {
  overflow-x: auto;
  border: 1px solid var(--border-main);
  border-radius: 0.5rem;
}

.quota-table {
  width: 100%;
  border-collapse: collapse;
  color: var(--text-main);
  font-size: var(--font-ui-xs);
  font-weight: 800;
  text-align: right;
}

.quota-table th,
.quota-table td {
  padding: 0.68rem 0.75rem;
  border-bottom: 1px solid var(--border-main);
  white-space: nowrap;
}

.quota-table th {
  background: color-mix(in srgb, var(--bg-main) 68%, transparent);
  color: var(--text-secondary);
  font-weight: 950;
}

.quota-table th:first-child,
.quota-table td:first-child {
  text-align: left;
}

.quota-table tbody tr:last-child td {
  border-bottom: 0;
}

.quota-table .patterns {
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
}

.threshold-table {
  table-layout: fixed;
}

.replay-cost-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  margin-top: 0.65rem;
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 800;
}

.replay-cost-row strong {
  color: var(--accent);
  font-size: var(--font-ui-sm);
}

.quota-notes {
  display: grid;
  grid-template-columns: auto 1fr 1fr;
  gap: 0.45rem 1rem;
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 750;
  line-height: 1.45;
}

.quota-notes h3 {
  grid-row: span 2;
  padding-right: 0.5rem;
}
</style>
