<template>
  <div class="page-root announcements-page">
    <header><Megaphone :size="24" /><h1>{{ $t('announcements.title') }}</h1></header>
    <div class="announcements-layout">
      <nav :aria-label="$t('announcements.history')">
        <h2>{{ $t('announcements.history') }}</h2>
        <button v-for="item in ANNOUNCEMENTS" :key="item.id" :class="{ selected: item.id === selected.id }"
          :aria-current="item.id === selected.id ? 'page' : undefined" @click="selectedId = item.id">
          <time :datetime="item.date">{{ item.date }}</time><span>{{ $t(item.titleKey) }}</span>
        </button>
      </nav>
      <article :key="selected.id">
        <time :datetime="selected.date">{{ selected.date }}</time>
        <h2>{{ $t(selected.titleKey) }}</h2>
        <p>{{ $t(selected.bodyKey) }}</p>
        <a v-if="selected.liveUrl" :href="selected.liveUrl" target="_blank" rel="noopener noreferrer" class="live-link">
          <Radio :size="19" />{{ selected.liveUrl }}<ExternalLink :size="16" />
        </a>
        <TokenSources v-if="selected.rewardCopyKey" :copy-key="selected.rewardCopyKey" />
        <button v-if="selected.rewardCopyKey" class="action-btn-small" @click="$emit('open-quota')"><CircleHelp :size="16" />{{ $t('billing.quotaGuide.open') }}</button>
      </article>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { CircleHelp, ExternalLink, Megaphone, Radio } from '@lucide/vue';
import TokenSources from '../billing/TokenSources.vue';
import { ANNOUNCEMENTS, findAnnouncement } from './catalog.js';
const props = defineProps({ requestedId: { type: String, default: '' } });
defineEmits(['open-quota']);
const selectedId = ref(props.requestedId);
const selected = computed(() => findAnnouncement(selectedId.value));
watch(() => props.requestedId, id => { selectedId.value = id; });
</script>

<style scoped>
.announcements-page { padding: 2rem; overflow: auto; color: var(--text-main); }
header { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 1.4rem; }
h1 { font-size: 1.5rem; font-weight: 900; }
.announcements-layout { display: grid; grid-template-columns: 15rem minmax(0, 1fr); gap: 2rem; max-width: 1240px; margin: auto; }
nav { border-right: 1px solid var(--border-main); padding-right: 1rem; }
nav h2 { font-size: var(--font-ui-sm); color: var(--text-secondary); margin-bottom: 0.75rem; }
nav button { display: grid; gap: 0.4rem; width: 100%; padding: 0.85rem; text-align: left; border-radius: 6px; border-left: 3px solid transparent; font-size: var(--font-ui-sm); }
nav button.selected { border-color: var(--accent); background: var(--bg-card); }
time { font-size: var(--font-ui-xs); color: var(--text-secondary); }
article { min-width: 0; padding-bottom: 2rem; }
article h2 { margin: 0.4rem 0 1rem; font-size: 1.4rem; font-weight: 900; }
article p { line-height: 1.75; font-size: var(--font-ui-sm); color: var(--text-secondary); }
.live-link { display: inline-flex; align-items: center; gap: 0.5rem; margin: 0.8rem 0 1.5rem; color: var(--accent); font-weight: 750; }
.live-link:hover { text-decoration: underline; }
article > button { width: fit-content; display: inline-flex; align-items: center; gap: 0.4rem; }
.token-sources { border-top: 1px solid var(--border-main); padding-top: 1.2rem; margin-bottom: 1.4rem; }
</style>
