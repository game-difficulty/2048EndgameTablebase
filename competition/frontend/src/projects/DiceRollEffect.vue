<template>
  <div v-if="visible" class="dice-curtain" aria-live="polite">
    <div class="die" :class="`face-${dice}`" :style="{'--dice-roll-duration': `${DICE_ROLL_DURATION}ms`}"><i v-for="dot in 9" :key="dot"></i></div>
    <strong>{{ $t('掷出 ') }}{{ dice }}{{ $t(' 点') }}</strong>
    <span>{{ $t(placement) }}</span>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import { DICE_REVEAL_DURATION, DICE_ROLL_DURATION } from './boardMotion.js';
const props = defineProps({dice: Number, rollKey: Number, moves: Number, enabled: Boolean});
const visible = ref(false);
const placement = computed(() => props.dice <= 3 ? '角位放置墙' : props.dice <= 5 ? '边位放置墙' : '中心位放置墙');
let timer;
// Moves deliberately are not a trigger: playing the first step must not cut
// short the curtain or restart its timer. Joining mid-game does not roll again.
watch(() => [props.enabled, props.dice, props.rollKey], () => {
  if (!props.enabled || !props.dice) { clearTimeout(timer); visible.value = false; return; }
  if (props.moves > 0) return;
  clearTimeout(timer);
  visible.value = true;
  timer = setTimeout(() => { visible.value = false; }, DICE_REVEAL_DURATION);
}, {immediate: true});
onBeforeUnmount(() => clearTimeout(timer));
</script>

<style scoped>
.dice-curtain{position:absolute;inset:0;z-index:20;display:grid;place-items:center;align-content:center;gap:10px;border-radius:12px;background:rgba(23,27,33,.9);color:#fff;text-align:center;pointer-events:none}
.die{display:grid;grid-template:repeat(3,18px)/repeat(3,18px);gap:5px;padding:18px;border-radius:14px;background:#f7f2e8;box-shadow:0 12px 30px rgba(0,0,0,.35);animation:dice-roll var(--dice-roll-duration) cubic-bezier(.2,.8,.2,1)}
.die i{width:12px;height:12px;border-radius:50%;background:transparent}
.face-1 i:nth-child(5),.face-2 i:nth-child(1),.face-2 i:nth-child(9),.face-3 i:nth-child(1),.face-3 i:nth-child(5),.face-3 i:nth-child(9),.face-4 i:nth-child(1),.face-4 i:nth-child(3),.face-4 i:nth-child(7),.face-4 i:nth-child(9),.face-5 i:nth-child(1),.face-5 i:nth-child(3),.face-5 i:nth-child(5),.face-5 i:nth-child(7),.face-5 i:nth-child(9),.face-6 i:nth-child(1),.face-6 i:nth-child(3),.face-6 i:nth-child(4),.face-6 i:nth-child(6),.face-6 i:nth-child(7),.face-6 i:nth-child(9){background:#40372f}
@keyframes dice-roll{0%{transform:translateY(-80px) rotate(-240deg) scale(.5);opacity:0}70%{transform:translateY(8px) rotate(18deg) scale(1.08)}100%{transform:none;opacity:1}}
</style>
