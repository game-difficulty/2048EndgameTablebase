<template>
  <Teleport defer :to="target" :disabled="!target">
    <div class="room-activity-entry" :class="[`${kind}-floating`, { docked: target }]">
      <button class="activity-dismiss" :aria-label="dismissLabel" @click="$emit('dismiss')"><X :size="14" /></button>
      <button class="activity-open" @click="$emit('open')">
        <slot /><b>{{ label }}</b><span>{{ caption }}</span>
      </button>
    </div>
  </Teleport>
</template>
<script setup>
import { X } from '@lucide/vue';
defineProps({ target: String, kind: String, label: String, caption: String, dismissLabel: String });
defineEmits(['open', 'dismiss']);
</script>
<style scoped>
.room-activity-entry { position:absolute;top:165px;left:8px;z-index:46;width:96px;text-align:center; }
.room-activity-entry.docked { position:relative;inset:auto; }
.lucky-floating { order:1; }.red-floating { order:2;top:325px; }.prediction-floating { order:3;top:485px; }
.activity-open { display:flex;flex-direction:column;align-items:center;gap:3px;width:100%;padding:8px 4px;border:1px solid #efbf67;background:var(--bg-main);color:var(--text-main);border-radius:8px;box-shadow:0 6px 20px #0003;cursor:pointer; }
.red-floating .activity-open { border-color:#d95448; }.prediction-floating .activity-open { border-color:var(--border-main); }
.activity-open :deep(img) { width:64px;height:64px;object-fit:contain; }.activity-open :deep(svg) { width:54px;height:60px; }
.activity-open b { font-size:11px; }.activity-open span { font:700 12px ui-monospace,monospace; }
.activity-dismiss { position:absolute;right:-7px;top:-7px;z-index:1;display:grid;place-items:center;padding:3px;border-radius:50%;background:var(--bg-main);border:1px solid var(--border-main);color:var(--text-main);cursor:pointer; }
</style>
