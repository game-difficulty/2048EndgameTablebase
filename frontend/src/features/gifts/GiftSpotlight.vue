<template>
  <Teleport to="body">
    <div class="gift-spotlight" aria-hidden="true" :style="{ '--spotlight-duration': `${SPOTLIGHT_DURATION}ms` }">
      <div class="spotlight-scene">
        <div class="spotlight-art" :class="{ wide: event.gift_id === 'rip' }"><GiftAnimation :id="event.gift_id" /></div>
        <strong class="spotlight-title">{{ name }} <span>×{{ event.combo_count }}</span></strong>
        <span class="spotlight-sender">{{ event.actor?.name }}</span>
      </div>
    </div>
  </Teleport>
</template>
<script setup>
import GiftAnimation from './GiftAnimation.vue';
import { SPOTLIGHT_DURATION } from './giftSpotlights.js';
defineProps({ event: { type: Object, required: true }, name: String });
</script>
<style scoped>
.gift-spotlight { position:fixed;inset:0;z-index:110;display:grid;place-items:center;pointer-events:none;background:#050911c9;animation:spotlight-fade var(--spotlight-duration) linear both; }
.spotlight-scene { display:flex;flex-direction:column;align-items:center;gap:12px;width:480px;max-width:100%;text-align:center;color:#fff;padding:24px;animation:spotlight-arrive .45s ease-out both; }
.spotlight-art { width:320px;height:260px;display:grid;place-items:center;filter:drop-shadow(0 10px 24px #0006); }
.spotlight-art :deep(.gift-motion) { transform:scale(5); }
.spotlight-art.wide :deep(.gift-motion) { transform:scale(3); }
.spotlight-title { font-size:30px;line-height:1.3;overflow-wrap:anywhere; }.spotlight-title span { color:#ffe2a0;white-space:nowrap; }
.spotlight-sender { font-size:18px;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#dae4ec; }
@keyframes spotlight-fade { 0% { opacity:0; } 8%,85% { opacity:1; } 100% { opacity:0; } }
@keyframes spotlight-arrive { from { transform:translateY(14px) scale(.92); } to { transform:none; } }
@media(prefers-reduced-motion:reduce) { .gift-spotlight { display:none;animation:none; } }
</style>
