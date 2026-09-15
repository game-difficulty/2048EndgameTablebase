<template>
  <div v-if="ceremony" :class="['gift-ceremony', `ceremony-${id}`]" aria-hidden="true">
    <div class="ceremony-art">
      <template v-if="id === '2048'">
        <div class="merge-four"><i v-for="n in 4" :key="n" class="scene-tile">512</i></div>
        <div class="merge-two"><i class="scene-tile">1024</i><i class="scene-tile">1024</i></div>
        <i class="scene-tile merge-result">2048</i>
      </template>
      <template v-else-if="id === 'legend'">
        <div class="legend-pair"><i class="scene-tile black-tile">32768</i><i class="scene-tile black-tile">32768</i></div>
        <i class="scene-tile legend-result">65536</i>
        <span class="legend-rays"></span><span class="legend-laurel">❮ ❮</span><span class="legend-laurel right">❯ ❯</span>
      </template>
      <template v-else-if="id === 'crown'">
        <i class="scene-tile black-tile crowned-tile">32768</i><Crown class="falling-crown" :size="62" :stroke-width="1.5" />
      </template>
      <template v-else-if="id === 'final'">
        <i class="scene-tile final-tile">1024</i><span class="final-stamp">FINAL PHASE</span>
      </template>
    </div>
    <strong class="ceremony-caption">{{ name }}</strong>
  </div>
  <span v-else :class="['gift-motion', `motion-${id}`]" aria-hidden="true">
    <img v-if="id === 'rip'" src="/live-gifts/rip-motion.gif" width="240" height="100" alt="" />
    <svg v-else-if="id === 'moai'" viewBox="0 0 64 64" class="moai-cutout-motion">
      <defs><filter :id="matteId" color-interpolation-filters="sRGB" x="0" y="0" width="100%" height="100%">
        <feColorMatrix type="matrix" values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  -1.417 -4.768 -0.481 0 6.533" />
      </filter></defs>
      <image :href="giftAsset('moai-motion')" width="64" height="64" :filter="`url(#${matteId})`" />
    </svg>
    <img v-else-if="['button','tea'].includes(id)" :src="giftAsset(`${id}-motion`)" width="64" height="64" alt="" />
    <template v-else-if="id === 'whale'">
      <GiftIcon id="whale" /><span class="spout"><i v-for="n in 3" :key="n"></i></span>
    </template>
    <template v-else-if="id === 'chicken'">
      <img class="chicken-a" :src="giftAsset('chicken')" width="64" height="64" alt="" />
      <img class="chicken-b" :src="giftAsset('chicken-alt')" width="64" height="64" alt="" />
    </template>
    <svg v-else-if="id === 'serious'" viewBox="0 0 100 100" fill="none">
      <ellipse cx="50" cy="89" rx="43" ry="4" fill="#786443" opacity=".3" />
      <g stroke="#d89026" stroke-width="6" stroke-linecap="round"><path class="leg-left" d="m45 69-8 15-10 2" /><path class="leg-right" d="m55 69 8 15 10 2" /></g>
      <g class="split-body">
        <path d="M28 51 9 35l-3-9m65 25 19-16 4-9" stroke="#e7c877" stroke-width="9" stroke-linecap="round" />
        <ellipse cx="50" cy="56" rx="24" ry="29" fill="#f5d881" /><circle cx="50" cy="30" r="21" fill="#ffe59c" />
        <path d="m38 16 5-9 6 6 5-8 6 10" fill="#ffe59c" />
        <circle cx="42" cy="28" r="3" fill="#272320" /><circle cx="58" cy="28" r="3" fill="#272320" />
        <path d="m44 37 6-5 6 5-6 5Z" fill="#d68723" /><path d="m35 23 11 3m8 0 11-3" stroke="#8e7030" stroke-width="2" />
      </g>
    </svg>
  </span>
</template>
<script setup>
import { Crown } from '@lucide/vue';
import { useId } from 'vue';
import GiftIcon from './GiftIcon.vue';
import { giftAsset } from './giftArtwork.js';
defineProps({ id: String, name: String, ceremony: Boolean });
// Remove the animated source's white matte without replacing or retiming its frames.
const matteId = `gift-matte-${useId()}`;
</script>
<style scoped>
.gift-motion { position:relative;display:inline-flex;align-items:center;justify-content:center;width:46px;height:46px;flex-shrink:0; }
.gift-motion > img,.gift-motion > svg { width:100%;height:100%;object-fit:contain;border-radius:4px; }
.motion-rip { width:110px; }
.chicken-a,.chicken-b { position:absolute;inset:0; }
.chicken-a { animation:pose-a .9s steps(1,end) 4; }.chicken-b { opacity:0;animation:pose-b .9s steps(1,end) 4; }
@keyframes pose-a { 50% { opacity:0; } } @keyframes pose-b { 50% { opacity:1; } }
.motion-whale { padding-top:8px; }.motion-whale :deep(.gift-icon) { animation:whale-bob 1s ease-in-out 3; }
.spout { position:absolute;top:7px;left:24px; }.spout i { position:absolute;width:3px;height:6px;border-radius:50%;background:#57c9f4;animation:spout 1s ease-out 3; }
.spout i:nth-child(2) { --dx:-8px;animation-delay:.1s; }.spout i:nth-child(3) { --dx:8px;animation-delay:.2s; }
@keyframes spout { from { transform:translate(0,4px);opacity:0; } 35% { opacity:1; } to { transform:translate(var(--dx,0px),-15px);opacity:0; } }
@keyframes whale-bob { 50% { transform:translateY(3px) rotate(-5deg); } }
.leg-left,.leg-right { transform-box:view-box;transform-origin:50px 69px;animation:split-left 1.6s ease-in-out 2; }.leg-right { animation-name:split-right; }
.split-body { animation:body-drop 1.6s ease-in-out 2; }
@keyframes split-left { 40%,70% { transform:rotate(42deg); } } @keyframes split-right { 40%,70% { transform:rotate(-42deg); } }
@keyframes body-drop { 40%,70% { transform:translateY(9px); } }
.gift-ceremony { position:absolute;z-index:25;right:0;top:76px;width:280px;height:174px;border:1px solid #d2b463;background:#171c28;color:#ffe3a2;border-radius:8px;box-shadow:0 8px 25px #0004;pointer-events:none;overflow:hidden;animation:ceremony-in .3s ease-out; }
.ceremony-art { position:relative;height:138px;display:flex;align-items:center;justify-content:center; }
.ceremony-caption { display:block;text-align:center;font-size:14px;line-height:24px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:0 10px; }
.scene-tile { display:flex;align-items:center;justify-content:center;flex-shrink:0;width:68px;height:68px;border-radius:7px;background:#e4bd44;color:#fff9e5;border:1px solid #ffe698;font-style:normal;font-family:Arial,sans-serif;font-weight:800;font-size:22px;font-variant-numeric:tabular-nums; }
.merge-four,.merge-two,.legend-pair { display:flex;gap:8px;position:absolute; }.merge-four .scene-tile { width:46px;height:46px;font-size:16px;background:#d3b03c; }
.merge-four { animation:four-merge 1.05s ease-in forwards; }.merge-four .scene-tile:nth-child(1) { --shift:27px; }.merge-four .scene-tile:nth-child(2) { --shift:-27px; }.merge-four .scene-tile:nth-child(3) { --shift:27px; }.merge-four .scene-tile:nth-child(4) { --shift:-27px; }
.merge-four .scene-tile { animation:pair-shift 1s ease-in forwards; }
.merge-two { opacity:0;animation:two-merge 2.1s ease-in forwards; }.merge-two .scene-tile { --shift:38px;animation:pair-shift 1s 1.05s ease-in forwards; }.merge-two .scene-tile:last-child { --shift:-38px; }
.merge-result { opacity:0;animation:reveal-tile .65s 2.05s ease-out forwards;box-shadow:0 0 22px #e8c54a70; }
.black-tile { background:#0d1016;border-color:#e3c17a;font-size:19px; }.crowned-tile { transform:translateY(17px); }
.falling-crown { position:absolute;left:109px;top:12px;fill:#e8b947;color:#ffe4a2;animation:crown-drop 1.3s cubic-bezier(.2,.6,.3,1) both; }
.final-tile { animation:stamp-impact .3s 1s ease-out; }.final-stamp { position:absolute;padding:6px 10px;bottom:21px;border:3px double #de7277;border-radius:4px;color:#ffaaa9;background:#281d25;font-size:18px;font-weight:900;transform:rotate(-12deg);animation:stamp-land 1.1s ease-out both; }
.legend-pair { animation:four-merge 1.5s ease-in forwards; }.legend-pair .scene-tile { --shift:38px;animation:pair-shift 1.5s ease-in forwards; }.legend-pair .scene-tile:last-child { --shift:-38px; }
.legend-result { background:#563778;border-color:#d9b7ff;opacity:0;font-size:19px;animation:reveal-tile .65s 1.5s ease-out forwards;z-index:1; }
.legend-rays { position:absolute;inset:25px 70px;border:1px solid #c5a7f1;border-radius:8px;opacity:0;animation:legend-light 1.6s 1.5s ease-out forwards; }
.legend-laurel { position:absolute;left:62px;color:#eac776;font-size:22px;opacity:0;animation:laurel 1s 1.7s ease-out forwards; }.legend-laurel.right { left:auto;right:62px; }
@keyframes ceremony-in { from { opacity:0;transform:translateY(-6px); } }
@keyframes pair-shift { 65%,100% { transform:translateX(var(--shift)); } }
@keyframes four-merge { 85% { opacity:1; } to { opacity:0; } }
@keyframes two-merge { 0%,49% { opacity:0; } 50%,94% { opacity:1; } to { opacity:0; } }
@keyframes reveal-tile { from { opacity:0;transform:scale(.7); } 55% { opacity:1;transform:scale(1.12); } to { opacity:1;transform:scale(1); } }
@keyframes crown-drop { from { transform:translateY(-85px) rotate(-18deg); } 70% { transform:translateY(4px) rotate(3deg); } to { transform:none; } }
@keyframes stamp-land { 0%,60% { opacity:0;transform:scale(2) rotate(-18deg); } to { opacity:1;transform:scale(1) rotate(-12deg); } }
@keyframes stamp-impact { 50% { transform:translateY(3px); } }
@keyframes legend-light { 15% { opacity:1; } to { opacity:0;transform:scale(1.7); } }
@keyframes laurel { to { opacity:1; } }
</style>
