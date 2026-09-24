<template>
  <span :class="['gift-icon', `gift-${id}`]" aria-hidden="true">
    <IiiGiftArtwork v-if="id === 'iii'" />
    <img v-else-if="referenceArtwork[id]" :src="giftAsset(referenceArtwork[id])" alt="" loading="lazy" decoding="async" width="64" height="64" />
    <component v-else-if="legacy[id]" :is="legacy[id]" :size="size" :stroke-width="1.8" />
    <svg v-else viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
      <g v-if="id === 'heart'">
        <path d="M40 72 13 47C-3 31 9 10 25 14c7 1 12 6 15 11 4-7 10-11 17-11 17 0 26 19 11 34Z" fill="#b52b61" />
        <path d="M40 67 14 43C1 30 12 10 27 16c6 2 10 6 13 12 4-7 8-11 15-12 17-3 24 14 12 27Z" fill="#f06495" stroke="#cd366d" stroke-width="1.5" />
        <path d="m40 66-7-27 33 4Z" fill="#d9447b" opacity=".55" />
        <path d="M14 30c0-6 5-10 11-8M14 36l1 3" stroke="#ffe4ef" stroke-width="4" stroke-linecap="round" />
        <path d="M49 23c4-4 9-4 12-1" stroke="#ffb5cd" stroke-width="3" stroke-linecap="round" />
        <path d="m57 58 6-3 7 2-4 8-5-2-4 5-4-7Z" fill="#ffe0a0" stroke="#cf9442" stroke-width="1" />
        <path d="m63 54-2 15" stroke="#c18339" stroke-width="1.5" />
        <path d="m64 3 2 5 5 2-5 2-2 5-2-5-5-2 5-2Z" fill="#f5cb6b" />
        <path d="m10 59 2 4 4 1-4 2-2 4-1-4-4-2 4-1Z" fill="#ee9dbc" />
      </g>
      <g v-else-if="id === 'flowers'">
        <path d="M39 60 22 27m17 33 4-43m-4 43 18-29" stroke="#338873" stroke-width="3" />
        <path d="M30 44C16 46 12 38 10 30c12 0 20 5 20 14ZM47 45c12 0 19-6 22-16-13-1-21 5-22 16Z" fill="#62b5a0" />
        <path d="m17 35 12 8m32-7-13 8" stroke="#25866d" stroke-width="1.5" />
        <path d="m12 39 27 35 27-35-17 7-12-7-10 7Z" fill="#f9e9d6" stroke="#dbb595" stroke-width="1.5" />
        <path d="m12 39 27 35-6-23Zm54 0L39 74l7-24Z" fill="#a7d9cd" />
        <g v-for="(flower, index) in [[24,28,'#ed7aab'],[42,19,'#b895eb'],[57,32,'#edab52']]" :key="index" :transform="`translate(${flower[0]} ${flower[1]})`">
          <ellipse v-for="angle in [0,60,120]" :key="angle" rx="7" ry="15" :transform="`rotate(${angle})`" :fill="flower[2]" />
          <circle r="7" fill="#fff2a7" />
          <path d="M-4-4c8-5 13 5 6 9-6 3-11-5-5-7 3-1 4 3 1 3" stroke="#c58059" stroke-width="1.7" stroke-linecap="round" />
          <path d="M-10-4c0-3 2-6 4-7" stroke="#fff8eb" stroke-opacity=".7" stroke-width="2" stroke-linecap="round" />
        </g>
        <path d="M38 59c-15-13-20 2-1 5-5 2-7 5-8 10l10-6 7 8-2-12c18-3 14-17-6-5Z" fill="#d9668a" stroke="#a73b62" stroke-width="1.5" />
        <circle cx="39" cy="61" r="3" fill="#ffafc3" />
      </g>
      <g v-else-if="id === 'dealer'">
        <g transform="rotate(13 53 27)"><rect x="36" y="5" width="32" height="42" rx="4" fill="#fff5dc" stroke="#ccaa64" stroke-width="1.5" /><rect x="40" y="9" width="24" height="34" rx="2" stroke="#e4cea0" /><text x="52" y="28" dominant-baseline="central" font-size="24" fill="#297e69">2</text><path d="m41 13 2-2 2 2-2 2ZM59 37l2-2 2 2-2 2Z" fill="#bc9860" /></g>
        <path d="m10 54 10-11c4-5 8-6 13-7l9-12c3-4 8-1 6 3l-7 12 14-6c5-2 7 3 3 6l-8 5 10-3c5-1 7 4 2 6l-12 6-12 4-14 13Z" fill="#cadbe4" stroke="#657c8d" stroke-width="1.8" stroke-linejoin="round" />
        <path d="m12 52 10-10 12-4 10-13c1-2 4-1 3 1l-8 15 8 1-10 10-12 7-6 5Z" fill="#f4f8f7" />
        <path d="m40 43 11-6m-8 11 13-5m-8 10 9-4M22 44l6 2M18 49l5 3" stroke="#9fb6c3" stroke-width="1.4" stroke-linecap="round" />
        <path d="M31 40q5 3 5 8" stroke="#dce6ea" stroke-width="2" stroke-linecap="round" />
        <text x="30" y="50" dominant-baseline="central" transform="rotate(-20 30 50)" font-size="15" fill="#318574">2</text>
        <path d="m7 52 18 17-7 8L1 60Z" fill="#35475c" stroke="#71849a" stroke-width="1.5" />
        <path d="m10 53 15 14-3 3L7 56Z" fill="#e8ece9" /><circle cx="11" cy="66" r="2" fill="#edc66e" />
      </g>
      <g v-else-if="id === '666'" transform="rotate(-9 40 40)">
        <path d="m12 52-7 12m20-8-4 14m48-47 6-7M9 22l8 4" stroke="#f6c24d" stroke-width="3" stroke-linecap="round" />
        <text x="41" y="58" font-size="37" fill="#9c462d" stroke="#9c462d" stroke-width="4" paint-order="stroke">666</text>
        <text x="39" y="53" font-size="37" fill="#ffcc54" stroke="#fff0b0" stroke-width="2" paint-order="stroke">666</text>
        <path d="m38 6 3 7 8 1-6 5 1 8-6-4-7 4 2-8-6-5 8-1Z" fill="#ffe999" stroke="#dc963e" stroke-width="1.5" />
      </g>
      <g v-else-if="id === 'bug'">
        <path d="M7 20h57l10 10v31H16L6 50Z" fill="#212e38" stroke="#87d9b8" stroke-width="2" />
        <path d="M8 27h12m45 0h7M7 53h11m44 8h11M22 16h13m7 49h15" stroke="#4bd1b2" stroke-width="3" />
        <path d="m64 20 1 11 9-1" fill="#e4617b" />
        <text x="40" y="48" font-size="26" fill="#37d8dd" transform="translate(-2 1)">BUG!</text>
        <text x="40" y="46" font-size="26" fill="#ff8699">BUG!</text>
        <path d="M10 39h15m27 10h18" stroke="#212e38" stroke-width="2" />
        <path d="m10 8 7 7m-2-8-4 9m51 51 7 6" stroke="#e66c8d" stroke-width="2" stroke-linecap="round" />
      </g>
      <g v-else-if="id === 'klbm'">
        <path d="M9 21q0-10 11-10h40q12 0 12 12v28q0 10-12 10H40L25 73l2-12h-7Q9 61 9 51Z" fill="#48849a" />
        <path d="M6 17Q6 7 18 7h41q11 0 11 12v27q0 11-11 11H37L23 69l2-12h-7Q6 57 6 46Z" fill="#d4f0ed" stroke="#419daa" stroke-width="1.5" />
        <text x="38" y="33" font-size="23" fill="#2e677b">klbm</text>
        <text x="43" y="53" font-size="24" fill="#8b5dba" transform="rotate(7 43 45)">??</text>
        <path d="m71 7 4-4m-1 12h4" stroke="#8b5dba" stroke-width="2" stroke-linecap="round" />
      </g>
      <g v-else-if="id === 'rip'">
        <g v-for="(tile,index) in ['4','2','4']" :key="index" :transform="`translate(${2 + index * 26} 25)`">
          <rect width="24" height="31" rx="4" :fill="index === 1 ? '#228b72' : '#f0d6a3'" />
          <text x="12" y="23" font-size="23" :fill="index === 1 ? 'white' : '#73592e'">{{ tile }}</text>
        </g>
      </g>
      <g v-else>
        <rect x="9" y="15" width="62" height="60" rx="8" :fill="tileFill" :stroke="id === 'legend' ? '#c0a1ee' : '#cfad60'" stroke-width="2" />
        <path d="M15 28v-6h12" stroke="white" stroke-opacity=".45" stroke-width="2" stroke-linecap="round" />
        <text x="40" :y="['two','four'].includes(id) ? 45 : id === 'final' ? 46 : 53" :dominant-baseline="['two','four'].includes(id) ? 'central' : 'auto'" :font-size="['two','four'].includes(id) ? 36 : id === 'crown' || id === 'legend' ? 18 : 22" :fill="id === 'four' ? '#725e3e' : '#fffaf0'">{{ tileText }}</text>
        <g v-if="id === 'crown'">
          <path d="m19 23-4-17 14 8L40 2l11 12 14-8-4 17Z" fill="#f2c454" stroke="#ac7719" stroke-width="2" />
          <circle cx="40" cy="16" r="3" fill="#dc6685" />
        </g>
        <g v-if="id === 'legend'" stroke="#c8a2ee" stroke-width="2">
          <path d="m3 55 4-9-4-9m74 18-4-9 4-9M19 9h42M27 4h26" />
        </g>
        <g v-if="id === 'final'" transform="rotate(-10 40 61)">
          <rect x="5" y="52" width="70" height="17" rx="2" fill="#fff1d1" stroke="#a93f4a" stroke-width="2" />
          <text x="40" y="64" font-size="10" fill="#a93f4a">FINAL PHASE</text>
        </g>
      </g>
    </svg>
  </span>
</template>
<script setup>
import { computed } from 'vue';
import IiiGiftArtwork from './IiiGiftArtwork.vue';
import { Coffee, PartyPopper, Sparkles, Combine } from '@lucide/vue';
import { referenceArtwork, giftAsset } from './giftArtwork.js';
const props = defineProps({ id: String, size: { type: Number, default: 28 } });
const legacy = { coffee: Coffee, fireworks: PartyPopper, brilliant: Sparkles, merge: Combine };
const tileText = computed(() => ({ two: '2', four: '4', '2048': '2048', crown: '32768', legend: '65536', final: '1024' })[props.id] || '?');
const tileFill = computed(() => ({ two: '#269674', four: '#ecddbb', crown: '#16191e', legend: '#32225c', final: '#d9b035' })[props.id] || '#e4b73a');
</script>
<style scoped>
.gift-icon { width:40px; height:40px; display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; color:#efba54; }
.gift-icon > img,.gift-icon > svg { width:100%;height:100%;object-fit:contain; }
.gift-icon > img { border-radius:4px; }
text { font-family:Arial,sans-serif;font-weight:900;text-anchor:middle;letter-spacing:0; }
</style>
