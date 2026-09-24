<template>
  <svg viewBox="0 0 80 80" fill="none" :class="{ animated }" aria-hidden="true">
    <path d="m6 57 7-1M3 64l12-1m48-3 10 3M61 70h13" stroke="#8c9ba9" stroke-width="2" stroke-linecap="round" />
    <g v-for="(color,index) in ['#ffbc58','#ff718b','#62dce4']" :key="index" :transform="`translate(${12+index*23} ${index === 1 ? 12 : 16}) rotate(${index === 1 ? -9 : 9} 7 24)`">
      <g class="letter" :style="{ '--delay': `${index * 140}ms` }">
        <path d="M0 21h14v27H0z" fill="#26304d" stroke="#26304d" stroke-width="6" transform="translate(2 3)" stroke-linejoin="round" />
        <rect x="0" y="20" width="14" height="27" rx="3" :fill="color" stroke="#fff4de" stroke-width="2" />
        <path d="M4 25v13" stroke="white" stroke-opacity=".65" stroke-width="2" stroke-linecap="round" />
        <g class="dot">
          <rect x="0" y="2" width="14" height="12" rx="4" fill="#26304d" transform="translate(2 3)" />
          <rect x="0" y="1" width="14" height="12" rx="4" :fill="color" stroke="#fff4de" stroke-width="2" />
          <path d="m4 5 4-1" stroke="white" stroke-width="2" stroke-linecap="round" />
        </g>
      </g>
    </g>
    <path class="alarm" d="m28 7-3-5m13 4 1-5m12 7 4-4" stroke="#ffcf79" stroke-width="2.5" stroke-linecap="round" />
    <path class="sweat" d="M70 12s-5 6-5 9a5 5 0 0 0 10 0c0-3-5-9-5-9Z" fill="#80e7ef" stroke="#e5ffff" stroke-width="1.5" />
  </svg>
</template>
<script setup>
defineProps({ animated: Boolean });
</script>
<style scoped>
svg { width:100%;height:100%;overflow:visible; }
.letter,.dot { transform-box:fill-box;transform-origin:center bottom; }
.animated .letter { animation:slip-catch 1.6s var(--delay) 2 both; }
.animated .dot { animation:alarm-hop 1.6s var(--delay) 2 both; }
.animated .sweat { animation:sweat-drop 1.6s .25s 2 both; }
.animated .alarm { animation:alarm-flare 1.6s 2 both; }
@keyframes slip-catch { 0%,15%,100% { transform:none; } 30% { transform:translate(5px,3px) rotate(19deg); } 43% { transform:translate(-2px,-2px) rotate(-12deg); } 56% { transform:translate(1px,0) rotate(5deg); } 70% { transform:none; } }
@keyframes alarm-hop { 0%,18%,75%,100% { transform:none; } 34% { transform:translateY(-7px) rotate(-15deg); } 50% { transform:translateY(1px); } }
@keyframes sweat-drop { 0%,15% { opacity:0;transform:translateY(-4px); } 35%,55% { opacity:1; } 85%,100% { opacity:0;transform:translateY(7px); } }
@keyframes alarm-flare { 0%,100% { opacity:.4; } 30%,60% { opacity:1; } }
@media (prefers-reduced-motion:reduce) { .animated .letter,.animated .dot,.animated .sweat,.animated .alarm { animation:none; } }
</style>
