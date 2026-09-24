<template>
  <p class="chat-prediction-reward"><template v-for="(part,index) in parts" :key="index"><strong v-if="part.kind" :class="`reward-${part.kind}`">{{ part.text }}</strong><template v-else>{{ part.text }}</template></template></p>
</template>
<script setup>
import { computed } from 'vue';
import { predictionAnnouncementSegments } from './predictionAnnouncement.js';
const props = defineProps({ event: { type: Object, required: true }, lang: String });
const parts = computed(() => predictionAnnouncementSegments(props.event, props.lang));
</script>
<style scoped>
.chat-prediction-reward { border-left:2px solid #c9a45b;padding:6px 8px;background:var(--bg-card);overflow-wrap:anywhere; }
strong { font-weight:750; }
.reward-player,.reward-name { color:var(--accent); }
.reward-amount { color:var(--text-main); }
</style>
