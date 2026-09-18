<template>
  <component :is="entry.href ? 'a' : 'button'" class="auxiliary-entry"
    :href="entry.href" :target="entry.href ? '_blank' : undefined"
    :rel="entry.href ? 'noopener noreferrer' : undefined" :type="entry.href ? undefined : 'button'"
    :draggable="false" @click="activate">
    <component :is="icons[entry.icon]" :size="18" aria-hidden="true" />
    <span>{{ $t(entry.title) }}</span>
    <ExternalLink v-if="entry.href" :size="13" aria-hidden="true" />
  </component>
</template>
<script setup>
import { Radio, Clapperboard, Megaphone, Coins, CircleHelp, Ellipsis, ExternalLink } from '@lucide/vue';
const props = defineProps({ entry: { type: Object, required: true } });
const emit = defineEmits(['navigate']);
const icons = { Radio, Clapperboard, Megaphone, Coins, CircleHelp, Ellipsis };
function activate() { if (!props.entry.href) emit('navigate', props.entry); }
</script>
<style scoped>
.auxiliary-entry { display:inline-flex;align-items:center;gap:8px;flex-shrink:0;padding:10px 14px;border:0;background:transparent;color:var(--text-main);font:inherit;text-decoration:none;cursor:pointer;white-space:nowrap; }
.auxiliary-entry:hover { color:var(--accent);background:var(--bg-card); }
.auxiliary-entry:focus-visible { outline:2px solid var(--accent);outline-offset:2px; }
</style>
