<template>
  <span v-if="active" class="like-reaction" aria-hidden="true" @animationend.self="finish">
    <GiftIcon :id="active" />
  </span>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue';
import GiftIcon from './GiftIcon.vue';
import { giftAsset, referenceArtwork } from './giftArtwork.js';
import { likeReactionWeights, pickLikeReaction } from './likeReaction.js';

const active = ref(null);
let timer;
function finish() {
  clearTimeout(timer);
  active.value = null;
}
function play() {
  // Repeated clicks still count, but never restart or queue visual reactions.
  if (active.value) return;
  active.value = pickLikeReaction();
  timer = setTimeout(finish, 1700);
}
onMounted(() => {
  for (const [id] of likeReactionWeights) {
    if (referenceArtwork[id]) new Image().src = giftAsset(referenceArtwork[id]);
  }
});
onBeforeUnmount(finish);
defineExpose({ play });
</script>

<style scoped>
.like-reaction {
  position:absolute;
  bottom:calc(100% + 12px);
  right:16px;
  width:64px;
  height:64px;
  z-index:3;
  pointer-events:none;
  filter:drop-shadow(0 3px 6px #0003);
  animation:like-float 1.5s ease-in-out both;
}
.like-reaction .gift-icon { width:100%;height:100%; }
@keyframes like-float {
  0% { opacity:0;transform:translateY(10px) scale(.8); }
  20% { opacity:1;transform:translateY(0) scale(1); }
  65% { opacity:1;transform:translateY(-5px) scale(1); }
  100% { opacity:0;transform:translateY(-18px) scale(.94); }
}
@media (prefers-reduced-motion:reduce) {
  .like-reaction { animation-name:like-fade; }
}
@keyframes like-fade {
  0%,100% { opacity:0; }
  20%,65% { opacity:1; }
}
</style>
