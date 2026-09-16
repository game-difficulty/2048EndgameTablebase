<template>
  <aside v-if="!dismissed" class="announcement-banner" :aria-label="$t('announcements.title')">
    <Megaphone :size="18" aria-hidden="true" />
    <button class="announcement-link" :aria-label="$t(latestAnnouncement.summaryKey)" @click="$emit('navigate', latestAnnouncement.target)">
      <span class="announcement-track" aria-hidden="true">
        <span>{{ $t(latestAnnouncement.summaryKey) }}</span>
        <span>{{ $t(latestAnnouncement.summaryKey) }}</span>
      </span>
    </button>
    <button class="announcement-close" :title="$t('announcements.dismiss')" :aria-label="$t('announcements.dismiss')" @click="dismiss">
      <X :size="18" />
    </button>
  </aside>
</template>

<script setup>
import { ref } from 'vue';
import { Megaphone, X } from '@lucide/vue';
import { latestAnnouncement, wasDismissed, dismissAnnouncement } from './catalog.js';
defineEmits(['navigate']);
const dismissed = ref(false);
try { dismissed.value = wasDismissed(window.localStorage, latestAnnouncement.id); } catch { /* Storage unavailable. */ }
function dismiss() {
  dismissed.value = true;
  try { dismissAnnouncement(window.localStorage, latestAnnouncement.id); } catch { /* Storage unavailable. */ }
}
</script>

<style scoped>
.announcement-banner { position: absolute; top: 3.7rem; left: 1.5rem; right: 1.5rem; z-index: 60; height: 2.25rem; display: flex; align-items: center; gap: 0.75rem; padding: 0 0.7rem; border: 1px solid var(--accent); border-radius: 6px; background: var(--bg-card); color: var(--accent); box-shadow: 0 4px 18px #0002; }
.announcement-link { min-width: 0; flex: 1; overflow: hidden; text-align: left; font-size: var(--font-ui-sm); font-weight: 750; }
.announcement-track { display: flex; width: max-content; animation: notice-scroll 28s linear infinite; }
.announcement-track > span { flex: none; padding-right: 5rem; }
.announcement-banner:hover .announcement-track, .announcement-banner:focus-within .announcement-track { animation-play-state: paused; }
.announcement-close { display: grid; place-items: center; width: 1.9rem; height: 1.9rem; flex: none; border-radius: 4px; }
.announcement-close:hover { background: var(--bg-main); }
button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
@keyframes notice-scroll { to { transform: translateX(-50%); } }
@media (prefers-reduced-motion: reduce) { .announcement-track { animation: none; white-space: normal; width: auto; } .announcement-track > span { padding: 0; } .announcement-track > span + span { display: none; } }
</style>
