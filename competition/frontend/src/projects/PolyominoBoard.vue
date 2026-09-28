<template>
  <div
    ref="root"
    class="poly-board"
    :style="{ '--rows': rows, '--cols': cols, '--board-slide-duration': `${BOARD_SLIDE_DURATION}ms`, '--board-pop-duration': `${BOARD_POP_DURATION}ms`, aspectRatio: `${cols} / ${rows}` }"
    tabindex="0"
    @pointerdown="pointerDown"
    @pointerup="pointerUp"
    @pointercancel="pointer = null"
  >
    <div v-for="index in rows * cols" :key="`cell-${index}`" class="poly-cell" :style="cellStyle(index - 1)" />
    <div
      v-for="tile in activeTiles"
      :key="tile.id"
      :class="['poly-tile', `value-${tile.value}`, { moving: tile.moving, instant: tile.instant }]"
      :style="tileStyle(tile)"
    >
      <span :class="['poly-art', { hidden: tile.hidden, pop: tile.pop && !tile.hidden, appear: tile.appear && !tile.hidden }]">
        <span v-for="cell in tile.cells" :key="cell" class="poly-piece" :style="partStyle(tile, cell)" />
        <span v-for="link in links(tile)" :key="link.key" class="poly-bridge" :style="link.style" />
        <strong class="poly-label" :style="labelStyle(tile)">{{ tile.value }}</strong>
      </span>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue';
import {
  BOARD_ANIMATION_DURATION,
  BOARD_MERGE_REVEAL_DELAY,
  BOARD_POP_DURATION,
  BOARD_SLIDE_DURATION,
  BOARD_SPAWN_REVEAL_DELAY,
} from './boardMotion.js';

const props = defineProps({
  snapshot: { type: Object, required: true },
  disabled: Boolean,
});
const emit = defineEmits(['move']);
const root = ref(null);
const rows = computed(() => Number(props.snapshot?.rows || 4));
const cols = computed(() => Number(props.snapshot?.cols || 4));
const activeTiles = ref([]);
const GAP = 2.25;
let pointer = null;
let timers = [];
let animationEpoch = 0;

function cellWidth() { return (100 - GAP * (cols.value + 1)) / cols.value; }
function cellHeight() { return (100 - GAP * (rows.value + 1)) / rows.value; }
function row(cell) { return Math.floor(cell / cols.value); }
function col(cell) { return cell % cols.value; }
function cellStyle(cell) {
  return {
    left: `${GAP + col(cell) * (cellWidth() + GAP)}%`,
    top: `${GAP + row(cell) * (cellHeight() + GAP)}%`,
    width: `${cellWidth()}%`, height: `${cellHeight()}%`,
  };
}
function bounds(tile) {
  const tileRows = tile.cells.map(row), tileCols = tile.cells.map(col);
  const minRow = Math.min(...tileRows), maxRow = Math.max(...tileRows);
  const minCol = Math.min(...tileCols), maxCol = Math.max(...tileCols);
  const width = (maxCol - minCol + 1) * cellWidth() + (maxCol - minCol) * GAP;
  const height = (maxRow - minRow + 1) * cellHeight() + (maxRow - minRow) * GAP;
  return { minRow, minCol, width, height };
}
function tileStyle(tile) {
  const box = bounds(tile);
  return {
    left: `${GAP + box.minCol * (cellWidth() + GAP)}%`,
    top: `${GAP + box.minRow * (cellHeight() + GAP)}%`,
    width: `${box.width}%`, height: `${box.height}%`,
  };
}
function partStyle(tile, cell) {
  const box = bounds(tile);
  const occupied = new Set(tile.cells);
  const left = col(cell) > 0 && occupied.has(cell - 1);
  const right = col(cell) + 1 < cols.value && occupied.has(cell + 1);
  const up = row(cell) > 0 && occupied.has(cell - cols.value);
  const down = row(cell) + 1 < rows.value && occupied.has(cell + cols.value);
  return {
    left: `${(col(cell) - box.minCol) * (cellWidth() + GAP) / box.width * 100}%`,
    top: `${(row(cell) - box.minRow) * (cellHeight() + GAP) / box.height * 100}%`,
    width: `${cellWidth() / box.width * 100}%`,
    height: `${cellHeight() / box.height * 100}%`,
    borderRadius: `${!up && !left ? 8 : 0}px ${!up && !right ? 8 : 0}px ${!down && !right ? 8 : 0}px ${!down && !left ? 8 : 0}px`,
  };
}
function links(tile) {
  const box = bounds(tile), occupied = new Set(tile.cells);
  const result = [];
  // Percentage math is rounded independently for each absolutely positioned
  // piece. Let bridges overlap their neighbors by 1 CSS pixel so no subpixel
  // hairline appears, while keeping the exposed concave edge unchanged.
  const overlap = (base, amount) => amount ? `calc(${base}% ${amount > 0 ? '+' : '-'} ${Math.abs(amount)}px)` : `${base}%`;
  for (const cell of tile.cells) {
    if (col(cell) + 1 < cols.value && occupied.has(cell + 1)) {
      const left = ((col(cell) - box.minCol) * (cellWidth() + GAP) + cellWidth()) / box.width * 100;
      result.push({
        key: `${cell}-right`,
        style: {
          left: overlap(left, -1),
          top: `${(row(cell) - box.minRow) * (cellHeight() + GAP) / box.height * 100}%`,
          width: overlap(GAP / box.width * 100, 2), height: `${cellHeight() / box.height * 100}%`,
        },
      });
    }
    if (row(cell) + 1 < rows.value && occupied.has(cell + cols.value)) {
      const top = ((row(cell) - box.minRow) * (cellHeight() + GAP) + cellHeight()) / box.height * 100;
      result.push({
        key: `${cell}-down`,
        style: {
          left: `${(col(cell) - box.minCol) * (cellWidth() + GAP) / box.width * 100}%`,
          top: overlap(top, -1),
          width: `${cellWidth() / box.width * 100}%`, height: overlap(GAP / box.height * 100, 2),
        },
      });
    }
  }
  for (let r = box.minRow; r < Math.max(...tile.cells.map(row)); r += 1) {
    for (let c = box.minCol; c < Math.max(...tile.cells.map(col)); c += 1) {
      const square = [r * cols.value + c, r * cols.value + c + 1, (r + 1) * cols.value + c, (r + 1) * cols.value + c + 1];
      // A three-cell L leaves this central square open. Fill the junction
      // only when all four surrounding cells belong to the same tile.
      if (!square.every(cell => occupied.has(cell))) continue;
      const left = ((c - box.minCol) * (cellWidth() + GAP) + cellWidth()) / box.width * 100;
      const top = ((r - box.minRow) * (cellHeight() + GAP) + cellHeight()) / box.height * 100;
      result.push({
        key: `${r}-${c}-corner`,
        style: {
          left: `${left}%`, top: `${top}%`,
          width: overlap(GAP / box.width * 100, 1),
          height: overlap(GAP / box.height * 100, 1),
        },
      });
    }
  }
  return result;
}
function labelStyle(tile) {
  const box = bounds(tile);
  let labelCells = tile.cells;
  if (tile.cells.length === 3) {
    const occupied = new Set(tile.cells);
    const joint = tile.cells.find(cell =>
      [cell - cols.value, cell + cols.value, col(cell) ? cell - 1 : -1, col(cell) + 1 < cols.value ? cell + 1 : -1]
        .filter(neighbor => occupied.has(neighbor)).length === 2);
    if (joint != null) labelCells = [joint];
  }
  const centerCol = labelCells.reduce((sum, cell) => sum + col(cell), 0) / labelCells.length;
  const centerRow = labelCells.reduce((sum, cell) => sum + row(cell), 0) / labelCells.length;
  return {
    left: `${((centerCol - box.minCol) * (cellWidth() + GAP) + cellWidth() / 2) / box.width * 100}%`,
    top: `${((centerRow - box.minRow) * (cellHeight() + GAP) + cellHeight() / 2) / box.height * 100}%`,
  };
}
function rawTiles(source = props.snapshot?.tiles || []) {
  return source.map(tile => ({ ...tile, cells: tile.cells.slice(), instant: true, hidden: false, moving: false, pop: false, appear: false }));
}
function clearTimers() { timers.forEach(clearTimeout); timers = []; }

watch(() => props.snapshot?.revision, async () => {
  const epoch = ++animationEpoch;
  clearTimers();
  const transition = props.snapshot?.transition;
  if (transition?.kind !== 'move') {
    activeTiles.value = rawTiles();
    return;
  }
  // Same interruption contract as the normal board: commit the previous frame
  // instantly, force layout, then animate the newest accepted move.
  activeTiles.value = rawTiles(transition.before);
  await nextTick();
  if (epoch !== animationEpoch) return;
  void root.value?.offsetHeight;
  const movements = new Map(transition.movements.map(item => [item.id, item]));
  const dying = new Set(transition.merges.flatMap(item => item.sources));
  const moving = activeTiles.value.map(tile => {
    const movement = movements.get(tile.id);
    return {
      ...tile,
      cells: movement?.to.slice() || tile.cells,
      instant: false,
      moving: Boolean(movement && movement.from.some((cell, index) => cell !== movement.to[index])),
      dying: dying.has(tile.id),
    };
  });
  const merged = transition.merges.map(item => ({
    ...item.tile, cells: item.tile.cells.slice(), instant: true,
    hidden: true, moving: false, pop: true, appear: false,
  }));
  const spawn = transition.spawn ? [{
    ...transition.spawn, cells: transition.spawn.cells.slice(), instant: true,
    hidden: true, moving: false, pop: false, appear: true,
  }] : [];
  activeTiles.value = [...moving, ...merged, ...spawn];
  timers.push(setTimeout(() => {
    if (epoch !== animationEpoch) return;
    activeTiles.value = activeTiles.value.filter(tile => !tile.dying).map(tile =>
      tile.pop ? { ...tile, hidden: false } : tile);
  }, BOARD_MERGE_REVEAL_DELAY));
  timers.push(setTimeout(() => {
    if (epoch !== animationEpoch) return;
    activeTiles.value = activeTiles.value.map(tile => tile.appear ? { ...tile, hidden: false } : tile);
  }, BOARD_SPAWN_REVEAL_DELAY));
  timers.push(setTimeout(() => {
    if (epoch !== animationEpoch) return;
    activeTiles.value = rawTiles();
  }, BOARD_ANIMATION_DURATION));
}, { immediate: true });

function pointerDown(event) {
  if (props.disabled || !event.isPrimary) return;
  pointer = { x: event.clientX, y: event.clientY, id: event.pointerId };
  event.currentTarget.setPointerCapture?.(event.pointerId);
}
function pointerUp(event) {
  if (!pointer || pointer.id !== event.pointerId || props.disabled) return;
  const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
  pointer = null;
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 18) return;
  emit('move', Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'right' : 'left') : (dy > 0 ? 'down' : 'up'));
}
onBeforeUnmount(() => { ++animationEpoch; clearTimers(); });
</script>

<style scoped>
.poly-board{position:relative;width:min(100%,620px);overflow:hidden;border-radius:12px;background:#a99d90;touch-action:none;outline:none;user-select:none;container-type:inline-size}
.poly-cell{position:absolute;border-radius:8px;background:#c8beb2}
.poly-tile{position:absolute;z-index:3;transition:left var(--board-slide-duration) ease-in-out,top var(--board-slide-duration) ease-in-out;pointer-events:none;--tile-color:#eee4da;color:#776e65}
.poly-tile.instant{transition:none}.poly-tile.moving{z-index:5}
.poly-art{position:absolute;inset:0;display:block;transition:opacity 0s}.poly-art.hidden{opacity:0}
.poly-piece,.poly-bridge{position:absolute;display:block;background:var(--tile-color)}.poly-piece{border-radius:8px}
.poly-label{position:absolute;z-index:1;transform:translate(-50%,-50%);font-size:clamp(16px,12cqw,44px);line-height:1;font-weight:800;white-space:nowrap}
.value-4{--tile-color:#ede0c8}.value-8{--tile-color:#f2b179;color:#f9f6f2}.value-16{--tile-color:#f59563;color:#f9f6f2}.value-32{--tile-color:#f67c5f;color:#f9f6f2}.value-64{--tile-color:#ef5b3c;color:#f9f6f2}.value-128{--tile-color:#edcf72;color:#f9f6f2}.value-256{--tile-color:#edcc61;color:#f9f6f2}
.poly-art.pop{animation:poly-pop var(--board-pop-duration) ease backwards}.poly-art.appear{animation:poly-appear var(--board-pop-duration) ease backwards}
@keyframes poly-pop{0%{transform:scale(1)}50%{transform:scale(1.2)}100%{transform:scale(1)}}
@keyframes poly-appear{0%{opacity:0;transform:scale(0)}100%{opacity:1;transform:scale(1)}}
</style>
