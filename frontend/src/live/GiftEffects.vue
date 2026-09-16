<template>
  <section class="gift-effects" :class="mode" :aria-label="t('直播动态', 'Live activity')">
    <div class="effect-lanes">
      <article v-for="event in mode === 'off' ? [] : events.active" :key="event.key" :class="['effect-banner', `tier-${event.tier || 0}`, { 'gold-supporter': liveSupporterLevel(event.actor) === 2, 'entrance-banner': event.type === 'entrance' }]">
        <div class="effect-copy"><LiveIdentity :actor="event.actor" :lang="lang" /><span v-if="event.type === 'entrance'">{{ t('欢迎荣耀赞助者来到直播间', 'Welcome to the stream, Gold Supporter') }}</span><span v-else>{{ t('送出', 'sent') }} {{ giftName(event.gift_id) }}</span></div>
        <GiftAnimation v-if="animations && giftAnimation(event) === 'supporter'" :id="event.gift_id" />
        <GiftIcon v-else-if="event.type === 'gift'" :id="event.gift_id" /><b v-if="event.type === 'gift'" class="combo">×{{ event.combo_count }}</b>
        <Crown v-else :size="22" class="entrance-crown" />
      </article>
    </div>
    <template v-if="animations">
      <GiftSpotlight v-if="spotlights.active" :key="spotlights.active.key" :event="spotlights.active" :name="giftName(spotlights.active.gift_id)" />
      <template v-else><GiftAnimation v-for="event in events.active.filter(item => giftAnimation(item) === 'ceremony')" :key="event.key" :id="event.gift_id" :name="giftName(event.gift_id)" ceremony /></template>
    </template>
  </section>
</template>
<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, watch } from 'vue';
import { Crown } from '@lucide/vue';
import LiveIdentity from './LiveIdentity.vue';
import { liveSupporterLevel } from './supporterIdentity.js';
import GiftIcon from './GiftIcon.vue';
import GiftAnimation from './GiftAnimation.vue';
import GiftSpotlight from './GiftSpotlight.vue';
import { GiftSpotlights } from './giftSpotlights.js';
import { giftAnimation } from './giftArtwork.js';
import { GiftEvents } from './giftEvents.js';
const props = defineProps({ lang: String, catalog: { type: Array, default: () => [] } });
const t = (zh,en) => props.lang === 'zh' ? zh : en;
const mode = defineModel('mode', { default: 'full' }), events = reactive(new GiftEvents());
const spotlights = reactive(new GiftSpotlights());
const reduced = ref(false);
const animations = computed(() => mode.value === 'full' && !reduced.value);
const giftName = id => props.catalog.find(item => item.id === id)?.[props.lang] || id;
let timer, motion;
function updateMotion() { reduced.value = motion.matches; }
function receive(event) {
  if (mode.value !== 'off') events.receive(event, Date.now(), 2);
  if (animations.value) spotlights.receive(event);
}
defineExpose({ receive });
watch(mode, value => { events.clear(); try { localStorage.setItem('live:effects', value); } catch {} });
watch(animations, () => spotlights.clear());
onMounted(() => {
  motion = matchMedia('(prefers-reduced-motion: reduce)');
  updateMotion(); motion.addEventListener('change', updateMotion);
  try { const saved = localStorage.getItem('live:effects'); if (['full','simple','off'].includes(saved)) mode.value = saved; else if (matchMedia('(prefers-reduced-motion: reduce)').matches) mode.value = 'simple'; } catch {}
  timer = setInterval(() => { events.tick(Date.now(), 2); spotlights.tick(); }, 100);
});
onUnmounted(() => { clearInterval(timer); motion?.removeEventListener('change', updateMotion); });
</script>
<style scoped>
.gift-effects { position:relative;height:88px; border-top:1px solid var(--border-main); margin-bottom:18px; display:flex; gap:16px; align-items:center; }
.effect-lanes { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; width:100%; min-width:0; }
.effect-banner { height:62px; border:1px solid var(--border-main); border-left:3px solid #39cfa0; background:var(--bg-card); border-radius:6px; padding:8px 12px; display:flex; align-items:center; gap:10px; min-width:0; }
.effect-banner.gold-supporter { position:relative;overflow:hidden;border-color:#b99240;border-left-width:3px; }
.gold-supporter::after { content:'';position:absolute;right:0;top:0;width:48px;height:15px;background:repeating-linear-gradient(135deg,transparent 0 5px,#d6b46145 5px 6px,transparent 6px 10px);pointer-events:none; }
.full .gold-supporter::before { content:'';position:absolute;top:0;left:8%;width:25%;height:1px;background:#f4d892;animation:gold-edge 1s ease-out;pointer-events:none; }
.entrance-banner { background:color-mix(in srgb,#d6b461 9%,var(--bg-main)); }
.full .effect-banner { animation:gift-arrive .25s ease-out; }
.tier-2 { border-left-color:#6acdf4; } .tier-3,.tier-4 { border-color:#dbb85a; }
.effect-copy { display:flex; flex-direction:column; gap:2px; min-width:0; flex:1; font-size:12px; }
.effect-copy strong,.effect-copy span { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.effect-copy > span:not(.live-identity) { color:var(--text-secondary);padding-left:31px; }.combo { color:#dbb85a; font-size:20px; font-variant-numeric:tabular-nums; }.entrance-crown { color:#dbb85a; }
@keyframes gold-edge { from { transform:translateX(-70%);opacity:0; } to { transform:translateX(190%);opacity:1; } }
@keyframes gift-arrive { from { opacity:0; transform:translateY(5px); } }
@media(prefers-reduced-motion:reduce) { .full .effect-banner,.full .gold-supporter::before { animation:none; } }
</style>
