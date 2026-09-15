<template>
  <span class="live-tile" :style="tileStyle" :title="value === 65536 ? '65k' : String(value)">{{ value }}</span>
</template>
<script setup>
import { computed } from 'vue';
import { liveTileColors } from './tilePalette.js';
const props = defineProps({ value: { type: Number, required: true } });
const tileStyle = computed(() => ({
  ...liveTileColors(props.value),
  fontSize: props.value >= 10000 ? '10px' : props.value >= 1000 ? '12px' : '14px',
}));
</script>
<style scoped>
.live-tile { display:inline-flex; align-items:center; justify-content:center; flex:0 0 42px; width:42px; height:42px; border-radius:3px; font-weight:700; line-height:1; letter-spacing:0; box-shadow:0 3px 8px #0002; }
</style>
