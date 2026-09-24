<template>
  <div class="human-board" :class="{ editable }" :style="{ '--cols': cols, '--rows': rows }"
       role="group" :aria-label="t(`${rows} 行 ${cols} 列棋盘`)" tabindex="0"
       @pointerdown="down" @pointerup="up" @pointercancel="pointer = null" @contextmenu="editable && $event.preventDefault()" @auxclick="editable && $event.preventDefault()">
    <button v-for="(value, index) in board" :key="index" type="button" class="tile"
      :class="{ empty: !value, big: value >= 1024, huge: value >= 32768 }"
      :style="tileStyle(0)" :tabindex="editable ? 0 : -1" :data-cell="index"
      :aria-label="t(`第 ${Math.floor(index / cols) + 1} 行第 ${index % cols + 1} 列，${value || '空格'}`)"
      @click="editable && $event.detail === 0 && $emit('cell', index, 0)"><span class="logical-tile-label">{{ hide32k && value === 32768 ? '' : value || '' }}</span></button>
    <div v-for="tile in activeTiles" :key="tile.id" class="moving-tile" :class="{ 'no-transition': tile.isInterrupting }" :style="position(tile)" aria-hidden="true">
      <div class="moving-tile-inner" :class="{ big: tile.value >= 1024, huge: tile.value >= 32768, 'anim-new': tile.isNew && !tile.isHidden, 'anim-merged': tile.isMerged && !tile.isHidden }" :style="{ ...tileStyle(tile.value), visibility: tile.isHidden ? 'hidden' : 'visible' }">{{ hide32k && tile.value === 32768 ? '' : tile.value }}</div>
    </div>
    <div v-if="$slots.overlay" class="board-overlay"><slot name="overlay" /></div>
  </div>
</template>
<script setup>
import { computed, shallowRef, watch } from 'vue';
import { tileStyle } from './appearance.js';
import { t } from './i18n.js';
import { humanBoardFrame } from './boardAnimation.js';
import { useBoardAnimation } from '../components/useBoardAnimation.js';
import { boardSwipeDirection } from '../components/boardPointerGesture.js';
const props = defineProps({ board: Array, rows: Number, cols: Number, transition: Object, editable: Boolean, hide32k: Boolean, touchButton: { type: Number, default: 0 } });
let revision = 0;
const frame = shallowRef(null);
const viewport = computed(() => ({ rows: props.rows, cols: props.cols, visibleIndices: props.board.map((_,i) => Math.floor(i / props.cols) * 4 + i % props.cols) }));
watch(() => [props.board, props.rows, props.cols, props.transition], () => {
  frame.value = humanBoardFrame(++revision, props.board, props.rows, props.cols, props.transition);
}, { immediate: true });
const { activeTiles } = useBoardAnimation({ get frame() { return frame.value; }, isVariant: false, animationDuration: 300 }, viewport, computed(() => `${props.rows}:${props.cols}`));
function position(tile) {
  return { left: `calc(var(--board-gap) + ${tile.col} * ((100% - (var(--cols) + 1) * var(--board-gap)) / var(--cols) + var(--board-gap)))`, top: `calc(var(--board-gap) + ${tile.row} * ((100% - (var(--rows) + 1) * var(--board-gap)) / var(--rows) + var(--board-gap)))` };
}
const emit = defineEmits(['move', 'cell']); let pointer = null;
function down(e) {
  const cell = e.target.closest('[data-cell]');
  if (!cell) return;
  if (props.editable && e.pointerType === 'mouse') { e.preventDefault(); emit('cell', Number(cell.dataset.cell), e.button); return; }
  if (e.isPrimary && e.button === 0) {
    if (e.pointerType === 'mouse') e.preventDefault();
    pointer = { x: e.clientX, y: e.clientY, id: e.pointerId, index: Number(cell.dataset.cell), type: e.pointerType };
    e.currentTarget.setPointerCapture?.(e.pointerId);
  }
}
function up(e) {
  if (!pointer || pointer.id !== e.pointerId) return;
  const start = pointer; pointer = null;
  if (e.currentTarget.hasPointerCapture?.(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  const bounds = e.currentTarget.getBoundingClientRect();
  const direction = boardSwipeDirection(e.clientX - start.x, e.clientY - start.y, Math.min(bounds.width, bounds.height));
  if (direction) emit('move', { up: 0, right: 1, down: 2, left: 3 }[direction]);
  else if (props.editable && start.type !== 'mouse') emit('cell', start.index, props.touchButton);
}
</script>
