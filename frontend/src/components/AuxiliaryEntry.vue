<template>
  <component :is="entry.href ? 'a' : 'button'" class="auxiliary-entry"
    :class="{ 'icon-only': entry.iconOnly, 'is-live': live }" :title="live ? $t('menu.liveOnline') : entry.iconOnly ? $t(entry.title) : undefined"
    :aria-label="entry.iconOnly ? $t(entry.title) : undefined"
    :href="entry.href" :target="entry.href ? '_blank' : undefined"
    :rel="entry.href ? 'noopener noreferrer' : undefined" :type="entry.href ? undefined : 'button'"
    :draggable="false" @click="activate">
    <component :is="icons[entry.icon]" :size="18" class="entry-icon" aria-hidden="true" />
    <span v-if="!entry.iconOnly">{{ $t(entry.title) }}</span>
    <ExternalLink v-if="entry.href && !entry.iconOnly" :size="12" class="external-mark" aria-hidden="true" />
  </component>
</template>
<script setup>
import { Radio, Clapperboard, Megaphone, Trophy, Coins, CircleHelp, Ellipsis, Mail, ExternalLink } from '@lucide/vue';
import Github from './GitHubIcon.vue';
const props = defineProps({ entry: { type: Object, required: true }, live: Boolean });
const emit = defineEmits(['navigate']);
const icons = { Radio, Clapperboard, Megaphone, Trophy, Coins, CircleHelp, Ellipsis, Mail, Github };
function activate() { if (!props.entry.href) emit('navigate', props.entry); }
</script>
<style scoped>
.auxiliary-entry { display:inline-flex;align-items:center;justify-content:center;gap:7px;flex-shrink:0;min-height:36px;padding:8px 12px;border:1px solid transparent;border-radius:6px;background:transparent;color:var(--text-secondary);font-family:inherit;font-size:var(--font-ui-sm,12px);font-weight:800;line-height:1.25;letter-spacing:0;text-decoration:none;cursor:pointer;white-space:nowrap;transition:color .15s,background-color .15s,border-color .15s; }
.auxiliary-entry:hover { color:var(--text-main);background:var(--bg-card);border-color:var(--border-main); }
.auxiliary-entry svg { flex-shrink:0; }
.external-mark { opacity:.65; }
.icon-only { width:36px;padding:8px; }
.top-live-entry { color:var(--text-main);font-weight:900; }
.is-live .entry-icon { color:#008577;animation:live-signal 2.4s ease-in-out infinite; }
[data-theme='dark'] .is-live .entry-icon { color:#34d9bd; }
@keyframes live-signal {
  0%,100% { opacity:1;filter:drop-shadow(0 0 0 transparent); }
  50% { opacity:.6;filter:drop-shadow(0 0 4px currentColor); }
}
@media (prefers-reduced-motion:reduce) { .is-live .entry-icon { animation:none; } }
.auxiliary-entry:focus-visible { outline:2px solid var(--accent);outline-offset:2px; }
</style>
