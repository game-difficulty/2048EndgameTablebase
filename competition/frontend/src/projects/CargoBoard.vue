<template>
  <div ref="stage" class="cargo-stage" tabindex="0" @pointerdown="pointerDown" @pointerup="pointerUp" @pointercancel="pointer = null">
    <div class="cargo-port cargo-entry" aria-label="入口">
      <span v-for="cell in 4" :key="`entry-${cell}`" :style="portCellStyle(cell - 1, true)" />
      <b>入口</b>
    </div>
    <TournamentBoard class="cargo-number-board" :snapshot="snapshot" :disabled="disabled" :tile-styles="tileStyles" :font-scale="fontScale" @move="emit('move', $event)" />
    <div class="cargo-port cargo-exit" aria-label="出口">
      <span v-for="cell in 2" :key="`exit-${cell}`" :style="portCellStyle(cell - 1, false)" />
      <b>出口 ↓</b>
    </div>
    <div v-if="displayCargo" :key="displayCargo.id" :class="['cargo-piece-box', { instant: cargoInstant }]" :style="cargoStyle(displayCargo)">
      <div class="cargo-art">
        <span v-for="(cell, index) in shapeCells(displayCargo)" :key="index" class="cargo-part" :style="partStyle(displayCargo, cell)" />
        <span v-for="(bridge, index) in shapeBridges(displayCargo)" :key="`bridge-${index}`" class="cargo-bridge" :style="bridge" />
        <strong :style="cargoLabelStyle(displayCargo)">{{ displayCargo.shape === 0 ? '▣' : '◆' }}</strong>
      </div>
    </div>
  </div>
</template>

<script setup>
import { nextTick, onBeforeUnmount, ref, watch } from 'vue';
import TournamentBoard from './TournamentBoard.vue';
import { CARGO_SHAPES } from './cargoEngine.js';
import { BOARD_SLIDE_DURATION } from './boardMotion.js';

const props = defineProps({
  snapshot: { type: Object, required: true },
  disabled: Boolean,
  tileStyles: { type: Object, default: () => ({}) },
  fontScale: { type: Number, default: 1 },
});
const emit = defineEmits(['move']);
const stage = ref(null);
const displayCargo = ref(null);
const cargoInstant = ref(true);
let pointer = null;
let revealTimer = null;
let animationEpoch = 0;

const GAP = 2.25;
const CELL = (100 - GAP * 5) / 4;
const PITCH = CELL + GAP;
const STAGE_HEIGHT = 175;
const BOX = CELL * 2 + GAP;
const partPosition = offset => offset ? (CELL + GAP) / BOX * 100 : 0;
const partSize = CELL / BOX * 100;
const gapStart = CELL / BOX * 100;
const gapSize = GAP / BOX * 100;

function topPercent(row) { return (50 + GAP + row * PITCH) / STAGE_HEIGHT * 100; }
function heightPercent(rows) { return (CELL * rows + GAP * (rows - 1)) / STAGE_HEIGHT * 100; }
function leftPercent(col) { return GAP + col * PITCH; }
function widthPercent(cols) { return CELL * cols + GAP * (cols - 1); }
function portCellStyle(index, entry) {
  const row = entry ? -2 + Math.floor(index / 2) : 4;
  const col = 1 + index % 2;
  return {
    left: `${leftPercent(col)}%`, top: `${topPercent(row)}%`,
    width: `${CELL}%`, height: `${heightPercent(1)}%`,
  };
}
function cargoStyle(cargo) {
  return {
    left: `${leftPercent(cargo.col)}%`, top: `${topPercent(cargo.row)}%`,
    width: `${widthPercent(2)}%`, height: `${heightPercent(2)}%`,
    '--board-slide-duration': `${BOARD_SLIDE_DURATION}ms`,
  };
}
function shapeCells(cargo) { return CARGO_SHAPES[cargo.shape]?.cells || []; }
function cargoLabelStyle(cargo) {
  if (cargo.shape === 0) return null;
  const cells = shapeCells(cargo);
  const joint = cells.find(([row, col]) => cells.filter(([otherRow, otherCol]) =>
    Math.abs(row - otherRow) + Math.abs(col - otherCol) === 1).length === 2);
  if (!joint) return null;
  return {
    left: `${(joint[1] * PITCH + CELL / 2) / BOX * 100}%`,
    top: `${(joint[0] * PITCH + CELL / 2) / BOX * 100}%`,
  };
}
function partStyle(cargo, [row, col]) {
  const cells = new Set(shapeCells(cargo).map(([r, c]) => `${r},${c}`));
  const has = (r, c) => cells.has(`${r},${c}`);
  return {
    left: `${partPosition(col)}%`, top: `${partPosition(row)}%`,
    width: `${partSize}%`, height: `${partSize}%`,
    borderRadius: `${!has(row - 1, col) && !has(row, col - 1) ? 9 : 0}px ${!has(row - 1, col) && !has(row, col + 1) ? 9 : 0}px ${!has(row + 1, col) && !has(row, col + 1) ? 9 : 0}px ${!has(row + 1, col) && !has(row, col - 1) ? 9 : 0}px`,
  };
}
function shapeBridges(cargo) {
  const cells = new Set(shapeCells(cargo).map(([row, col]) => `${row},${col}`));
  const result = [];
  for (const [row, col] of shapeCells(cargo)) {
    if (cells.has(`${row},${col + 1}`)) result.push({
      left: `calc(${gapStart}% - 1px)`, top: `${partPosition(row)}%`,
      width: `calc(${gapSize}% + 2px)`, height: `${partSize}%`,
    });
    if (cells.has(`${row + 1},${col}`)) result.push({
      left: `${partPosition(col)}%`, top: `calc(${gapStart}% - 1px)`,
      width: `${partSize}%`, height: `calc(${gapSize}% + 2px)`,
    });
  }
  // Only a complete 2×2 block needs the central gap filled. For an L shape
  // that same square is its open inner corner and must stay transparent.
  if (cells.size === 4) {
    result.push({
      left: `${gapStart}%`, top: `${gapStart}%`,
      width: `calc(${gapSize}% + 1px)`, height: `calc(${gapSize}% + 1px)`,
    });
  }
  return result;
}

watch(() => props.snapshot?.revision, async () => {
  const epoch = ++animationEpoch;
  clearTimeout(revealTimer);
  const transition = props.snapshot?.transition;
  if (transition?.kind !== 'move' || !transition.cargoMoved) {
    displayCargo.value = props.snapshot?.cargo || null;
    cargoInstant.value = true;
    return;
  }
  displayCargo.value = transition.cargoBefore;
  cargoInstant.value = true;
  await nextTick();
  if (epoch !== animationEpoch) return;
  void stage.value?.offsetHeight;
  cargoInstant.value = false;
  displayCargo.value = transition.cargoExit || props.snapshot.cargo;
  if (transition.cargoExit) {
    revealTimer = setTimeout(() => {
      if (epoch !== animationEpoch) return;
      displayCargo.value = props.snapshot.cargo;
      cargoInstant.value = true;
    }, BOARD_SLIDE_DURATION);
  }
}, { immediate: true });

function pointerDown(event) {
  if (props.disabled || !event.isPrimary || event.target.closest('.tournament-board')) return;
  pointer = { id: event.pointerId, x: event.clientX, y: event.clientY };
  event.currentTarget.setPointerCapture?.(event.pointerId);
}
function pointerUp(event) {
  if (!pointer || pointer.id !== event.pointerId || props.disabled) return;
  const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
  pointer = null;
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 18) return;
  emit('move', Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'right' : 'left') : (dy > 0 ? 'down' : 'up'));
}
onBeforeUnmount(() => { ++animationEpoch; clearTimeout(revealTimer); });
</script>

<style scoped>
.cargo-stage{position:relative;width:min(100%,500px);aspect-ratio:4/7;margin:auto;touch-action:none;user-select:none;outline:none;overflow:hidden}
.cargo-number-board{position:absolute;left:0;top:28.5714285714%;width:100%;max-width:none;z-index:1}
.cargo-port{position:absolute;inset:0;z-index:4;pointer-events:none}.cargo-port span{position:absolute;box-sizing:border-box;border:2px dashed #ab8e5c;border-radius:8px;background:transparent}
.cargo-port b{position:absolute;left:50%;transform:translateX(-50%);font-size:11px;color:#937443;letter-spacing:.14em;white-space:nowrap}
.cargo-entry b{top:1px}.cargo-exit b{bottom:1px}
.cargo-piece-box{position:absolute;z-index:3;pointer-events:none;transition:left var(--board-slide-duration) ease-in-out,top var(--board-slide-duration) ease-in-out}
.cargo-piece-box.instant{transition:none}
.cargo-art{position:absolute;inset:0;filter:drop-shadow(0 3px 4px rgba(29,48,58,.28));color:#f8fbfa}
.cargo-part,.cargo-bridge{position:absolute;display:block;background:#448d8a}.cargo-part{border-radius:9px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.22)}
.cargo-art strong{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);font-size:clamp(24px,4vw,42px);line-height:1;text-shadow:0 1px 2px rgba(0,0,0,.3)}
</style>
