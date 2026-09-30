<template>
  <article class="gift-sort-card" :class="{ dragging }" :data-sort-index="index">
    <span class="gift-rank">{{ index + 1 }}</span>
    <GiftIcon :id="gift.id" :size="42" />
    <strong>{{ gift[lang] }}</strong>
    <button class="gift-drag-handle" type="button"
      :aria-label="t(`拖动${gift[lang]}，当前位置 ${index + 1}`, `Move ${gift[lang]}, position ${index + 1}`)"
      @pointerdown="$emit('drag-start', $event)" @keydown="onKeydown"><GripVertical :size="17" /></button>
  </article>
</template>

<script setup>
import { GripVertical } from '@lucide/vue';
import GiftIcon from './GiftIcon.vue';
const props = defineProps({ gift: Object, lang: String, index: Number, dragging: Boolean });
const emit = defineEmits(['drag-start', 'move']);
const t = (zh, en) => props.lang === 'zh' ? zh : en;
function onKeydown(event) {
  const offsets = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -3, ArrowDown: 3 };
  if (!Object.hasOwn(offsets, event.key)) return;
  event.preventDefault();
  emit('move', offsets[event.key]);
}
</script>

<style scoped>
.gift-sort-card { position:relative;min-width:0;height:105px;display:grid;grid-template-rows:48px minmax(28px,auto) 24px;place-items:center;padding:5px 4px;border:1px solid var(--border-main);border-radius:6px;background:var(--bg-card);transition:opacity .12s,border-color .12s; }
.gift-sort-card.dragging { opacity:.3;border-color:var(--accent); }
.gift-rank { position:absolute;top:5px;left:6px;min-width:20px;padding:2px 4px;border-radius:9px;background:var(--bg-input);color:var(--text-secondary);font-size:10px;font-variant-numeric:tabular-nums;text-align:center; }
strong { align-self:start;max-width:100%;font-size:11px;line-height:14px;text-align:center;overflow-wrap:anywhere; }
.gift-drag-handle { width:38px;min-height:24px;padding:1px 8px;touch-action:none;cursor:grab;color:var(--text-secondary);background:var(--bg-input);border:1px solid var(--border-main); }
.gift-drag-handle:active { cursor:grabbing; }
.gift-drag-handle:focus-visible { outline:2px solid var(--accent);outline-offset:1px; }
</style>
