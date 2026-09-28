<template>
  <div
    ref="root"
    :class="['tournament-board', { mirror: mirrorPortals, irregular: irregularShape }]"
    :style="boardStyle"
    tabindex="0"
    @pointerdown="pointerDown"
    @pointerup="pointerUp"
    @pointercancel="pointer = null"
  >
    <div
      v-for="(value, index) in board"
      :key="`cell-${index}`"
      :class="['board-cell', { blocked: value === WALL }]"
      :style="cellPosition(index)"
    />
    <div
      v-for="tile in activeTiles"
      :key="tile.id"
      :class="['board-tile', tileClass(tile.value), {
        moving: tile.isAnimating,
        'no-transition': tile.isInterrupting,
        hidden: tile.isHidden,
        pop: tile.isMerged && !tile.isHidden,
        appear: tile.isNew && !tile.isHidden,
      }]"
      :style="tilePosition(tile)"
    ><span>{{ tileLabel(tile.value) }}</span></div>
    <TransitionGroup name="board-seal" tag="div" class="board-seals" aria-hidden="true">
      <div v-for="index in sealedCells" :key="`seal-${index}`" class="board-seal" :style="cellPosition(index)">
        <svg class="board-seal-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg>
      </div>
    </TransitionGroup>
    <div v-if="mirrorPortals" class="mirror-cross" aria-hidden="true"><i></i><b></b></div>
    <div v-if="mirrorPortals" class="portal-labels" aria-hidden="true"><span>↔</span><span>↕</span></div>
    <div v-if="diceReveal" class="board-dice-effect" aria-live="polite">
      <div class="board-die" :class="`face-${snapshot.dice}`"><i v-for="dot in 9" :key="dot"></i></div>
      <strong>{{ snapshot.dice }} 点</strong>
    </div>
    <slot name="overlay" />
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue';
import { ISLAND, WALL } from './engine.js';
import {
  BOARD_ANIMATION_DURATION,
  BOARD_MERGE_REVEAL_DELAY,
  BOARD_POP_DURATION,
  BOARD_SLIDE_DURATION,
  BOARD_SPAWN_REVEAL_DELAY,
} from './boardMotion.js';

const props = defineProps({
  snapshot: { type: Object, required: true },
  mirrorPortals: Boolean,
  irregularShape: Boolean,
  showDiceEffect: Boolean,
  sealedCells: { type: Array, default: () => [] },
  disabled: Boolean,
});
const emit = defineEmits(['move']);
const root = ref(null);
const board = computed(() => props.snapshot?.board || []);
const rows = computed(() => Number(props.snapshot?.rows || 4));
const cols = computed(() => Number(props.snapshot?.cols || 4));
const IRREGULAR_GAP_UNITS = 0.1;
const irregularGeometry = computed(() => {
  if (!props.irregularShape) return null;
  return {
    width: cols.value + IRREGULAR_GAP_UNITS * (cols.value + 1),
    height: rows.value + IRREGULAR_GAP_UNITS * (rows.value + 1),
  };
});
const boardStyle = computed(() => {
  const geometry = irregularGeometry.value;
  return {
    '--rows': rows.value,
    '--cols': cols.value,
    '--board-pop-duration': `${BOARD_POP_DURATION}ms`,
    '--cell-width': geometry ? `${100 / geometry.width}%` : undefined,
    '--cell-height': geometry ? `${100 / geometry.height}%` : undefined,
    aspectRatio: geometry ? `${geometry.width} / ${geometry.height}` : `${cols.value} / ${rows.value}`,
  };
});
const activeTiles = ref([]);
const diceReveal = ref(false);
let pointer = null;
let timers = [];
let animationEpoch = 0;
let tileIdCounter = 0;

function clearTimers() {
  timers.forEach(clearTimeout);
  timers = [];
}

function row(index) { return Math.floor(index / cols.value); }
function col(index) { return index % cols.value; }
function percentPosition(r, c) {
  const geometry = irregularGeometry.value;
  if (geometry) {
    return {
      left: `${100 * (IRREGULAR_GAP_UNITS + c * (1 + IRREGULAR_GAP_UNITS)) / geometry.width}%`,
      top: `${100 * (IRREGULAR_GAP_UNITS + r * (1 + IRREGULAR_GAP_UNITS)) / geometry.height}%`,
    };
  }
  const gap = 2.25;
  const width = (100 - gap * (cols.value + 1)) / cols.value;
  const height = (100 - gap * (rows.value + 1)) / rows.value;
  return { left: `${gap + c * (width + gap)}%`, top: `${gap + r * (height + gap)}%` };
}
function cellPosition(index) { return percentPosition(row(index), col(index)); }

function wraps(from, to, direction) {
  if (!props.mirrorPortals) return false;
  if (direction === 'left') return col(to) > col(from);
  if (direction === 'right') return col(to) < col(from);
  if (direction === 'up') return row(to) > row(from);
  return row(to) < row(from);
}

function isRenderableTile(value) { return value > 0 || value === ISLAND; }

function createTile(index, value, overrides = {}) {
  return {
    id: `tile-${tileIdCounter++}`,
    row: row(index),
    col: col(index),
    targetRow: row(index),
    targetCol: col(index),
    value,
    duration: BOARD_SLIDE_DURATION,
    isAnimating: false,
    isInterrupting: true,
    isDying: false,
    isMerged: false,
    isNew: false,
    isHidden: false,
    ...overrides,
  };
}

function syncToBoardRaw(sourceBoard = board.value) {
  activeTiles.value = sourceBoard.flatMap((value, index) => (
    isRenderableTile(value) ? [createTile(index, value)] : []
  ));
}

function fastForwardAnimations(isInterrupting = false) {
  activeTiles.value = activeTiles.value
    .filter(tile => !tile.isDying)
    .map(tile => ({
      ...tile,
      row: tile.targetRow,
      col: tile.targetCol,
      duration: BOARD_SLIDE_DURATION,
      isAnimating: false,
      isInterrupting,
      isMerged: false,
      isNew: false,
      isHidden: false,
    }));
}

function activeTilesMatch(sourceBoard) {
  const rendered = sourceBoard.map(value => value === WALL ? WALL : 0);
  for (const tile of activeTiles.value) rendered[tile.row * cols.value + tile.col] = tile.value;
  return rendered.length === sourceBoard.length && rendered.every((value, index) => value === sourceBoard[index]);
}

function wrappedWaypoints(movement, direction) {
  const horizontal = ['left', 'right'].includes(direction);
  const negative = ['left', 'up'].includes(direction);
  return {
    exitRow: horizontal ? row(movement.from) : negative ? -1 : rows.value,
    exitCol: horizontal ? (negative ? -1 : cols.value) : col(movement.from),
    enterRow: horizontal ? row(movement.to) : negative ? rows.value : -1,
    enterCol: horizontal ? (negative ? cols.value : -1) : col(movement.to),
  };
}

function revealMergedTiles() {
  activeTiles.value = activeTiles.value.map(tile => {
    if (tile.isDying) return { ...tile, isHidden: true };
    if (tile.isMerged) return { ...tile, isHidden: false };
    return tile;
  });
}

function revealAppearingTiles() {
  activeTiles.value = activeTiles.value.map(tile => tile.isNew ? { ...tile, isHidden: false } : tile);
}

watch(() => props.snapshot?.revision, async () => {
  const epoch = ++animationEpoch;
  clearTimers();
  diceReveal.value = false;
  const transition = props.snapshot?.transition;
  fastForwardAnimations(true);
  if (transition?.kind !== 'move') {
    syncToBoardRaw(board.value);
    return;
  }

  const sourceBoard = Array.isArray(transition.before) ? transition.before : board.value;
  if (!activeTilesMatch(sourceBoard)) syncToBoardRaw(sourceBoard);
  activeTiles.value.forEach(tile => {
    tile.isInterrupting = true;
    tile.isAnimating = false;
  });
  await nextTick();
  if (epoch !== animationEpoch) return;
  void root.value?.offsetHeight;

  const movements = new Map((transition.movements || []).map(item => [item.from, item]));
  const mergeDestinations = new Set();
  const nextTiles = [];
  const wrappedTiles = [];
  for (const tile of activeTiles.value) {
    const from = tile.row * cols.value + tile.col;
    const movement = movements.get(from);
    if (!movement) {
      tile.isInterrupting = false;
      nextTiles.push(tile);
      continue;
    }
    tile.targetRow = row(movement.to);
    tile.targetCol = col(movement.to);
    tile.isAnimating = movement.from !== movement.to;
    tile.isInterrupting = false;
    if (wraps(movement.from, movement.to, transition.direction)) {
      const waypoint = wrappedWaypoints(movement, transition.direction);
      tile.row = waypoint.exitRow;
      tile.col = waypoint.exitCol;
      tile.duration = BOARD_SLIDE_DURATION / 2;
      wrappedTiles.push({ id: tile.id, ...waypoint });
    } else {
      tile.row = tile.targetRow;
      tile.col = tile.targetCol;
      tile.duration = BOARD_SLIDE_DURATION;
    }
    if (movement.merged) {
      tile.isDying = true;
      if (!mergeDestinations.has(movement.to)) {
        mergeDestinations.add(movement.to);
        nextTiles.push(createTile(movement.to, board.value[movement.to], {
          isInterrupting: false,
          isMerged: true,
          isHidden: true,
        }));
      }
    }
    nextTiles.push(tile);
  }
  if (transition.spawn?.index != null && isRenderableTile(transition.spawn.value)) {
    nextTiles.push(createTile(transition.spawn.index, transition.spawn.value, {
      isInterrupting: false,
      isNew: true,
      isHidden: true,
    }));
  }
  activeTiles.value = nextTiles;

  if (wrappedTiles.length) {
    timers.push(setTimeout(async () => {
      if (epoch !== animationEpoch) return;
      for (const waypoint of wrappedTiles) {
        const tile = activeTiles.value.find(item => item.id === waypoint.id);
        if (!tile) continue;
        tile.isInterrupting = true;
        tile.row = waypoint.enterRow;
        tile.col = waypoint.enterCol;
      }
      await nextTick();
      if (epoch !== animationEpoch) return;
      void root.value?.offsetHeight;
      for (const waypoint of wrappedTiles) {
        const tile = activeTiles.value.find(item => item.id === waypoint.id);
        if (!tile) continue;
        tile.row = tile.targetRow;
        tile.col = tile.targetCol;
        tile.isInterrupting = false;
      }
    }, BOARD_SLIDE_DURATION / 2));
  }
  timers.push(setTimeout(() => {
    if (epoch !== animationEpoch) return;
    revealMergedTiles();
  }, BOARD_MERGE_REVEAL_DELAY));
  timers.push(setTimeout(() => {
    if (epoch !== animationEpoch) return;
    revealAppearingTiles();
  }, BOARD_SPAWN_REVEAL_DELAY));
  timers.push(setTimeout(() => {
    if (epoch !== animationEpoch) return;
    fastForwardAnimations(false);
  }, BOARD_ANIMATION_DURATION));
}, { immediate: true });

watch(() => props.snapshot?.dice, (value, previous) => {
  if (!props.showDiceEffect || !value || value === previous) return;
  diceReveal.value = true;
  timers.push(setTimeout(() => { diceReveal.value = false; }, 1400));
}, { immediate: true });

function tilePosition(tile) {
  return {
    ...percentPosition(tile.row, tile.col),
    transition: `left ${tile.duration}ms ease-in-out, top ${tile.duration}ms ease-in-out`,
  };
}

function tileClass(value) {
  if (value === WALL) return 'wall';
  if (value === ISLAND) return 'island';
  return `value-${Math.min(2048, Number(value) || 0)}`;
}
function tileLabel(value) { return value === WALL || value === ISLAND ? '' : value; }

function pointerDown(event) {
  if (props.disabled || !event.isPrimary) return;
  pointer = { x: event.clientX, y: event.clientY, id: event.pointerId };
  event.currentTarget.setPointerCapture?.(event.pointerId);
}
function pointerUp(event) {
  if (!pointer || pointer.id !== event.pointerId || props.disabled) return;
  const dx = event.clientX - pointer.x;
  const dy = event.clientY - pointer.y;
  pointer = null;
  if (Math.max(Math.abs(dx), Math.abs(dy)) < 18) return;
  emit('move', Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'right' : 'left') : (dy > 0 ? 'down' : 'up'));
}
onBeforeUnmount(() => { animationEpoch += 1; clearTimers(); });
</script>

<style scoped>
.tournament-board{--gap:2.25%;position:relative;width:min(100%,620px);aspect-ratio:1;margin:auto;overflow:hidden;border-radius:12px;background:#a99d90;touch-action:none;outline:none}.board-cell{float:left;width:calc((100% - (var(--cols) + 1) * var(--gap))/var(--cols));height:calc((100% - (var(--rows) + 1) * var(--gap))/var(--rows));margin:var(--gap) 0 0 var(--gap);border-radius:8px;background:#c8beb2}.tournament-board.irregular{background:transparent}.tournament-board.irregular .board-cell{background:#c8beb2;box-shadow:0 0 0 2px #a99d90}.tournament-board.irregular .board-cell.blocked{visibility:hidden}.board-tile{position:absolute;z-index:3;width:calc((100% - (var(--cols) + 1) * var(--gap))/var(--cols));height:calc((100% - (var(--rows) + 1) * var(--gap))/var(--rows));display:grid;place-items:center;border-radius:8px;background:#eee4da;color:#776e65;font-size:clamp(20px,6vw,48px);font-weight:800;line-height:1;box-shadow:inset 0 0 0 1px rgba(255,255,255,.15)}.board-tile.moving{z-index:5;pointer-events:none}.board-tile.wall{background:repeating-linear-gradient(135deg,#4d5662 0 8px,#424a55 8px 16px);box-shadow:inset 0 0 0 2px #697482}.board-tile.island{background:#172f3d url('https://2048tables.online/minigames-assets/portal.png?v=minigames-img-20260710b') center/cover no-repeat;box-shadow:0 0 10px rgba(56,189,248,.22)}.value-4{background:#ede0c8}.value-8{color:#f9f6f2;background:#f2b179}.value-16{color:#f9f6f2;background:#f59563}.value-32{color:#f9f6f2;background:#f67c5f}.value-64{color:#f9f6f2;background:#ef5b3c}.value-128{color:#f9f6f2;background:#edcf72;font-size:clamp(17px,5vw,40px)}.value-256{color:#f9f6f2;background:#edcc61;font-size:clamp(17px,5vw,40px)}.value-512{color:#f9f6f2;background:#edc850;font-size:clamp(17px,5vw,40px)}.value-1024,.value-2048{color:#f9f6f2;background:#edc53f;font-size:clamp(14px,4vw,34px)}.pop{animation:tile-pop 200ms ease}.appear{animation:tile-appear 200ms ease backwards}.mirror-cross{position:absolute;inset:0;z-index:2;pointer-events:none}.mirror-cross i,.mirror-cross b{position:absolute;display:block;background:#6d6258;box-shadow:0 0 0 2px rgba(42,35,30,.2)}.mirror-cross i{left:calc(50% - var(--gap)/2);top:0;width:var(--gap);height:100%}.mirror-cross b{left:0;top:calc(50% - var(--gap)/2);width:100%;height:var(--gap)}.portal-labels span{position:absolute;z-index:6;color:#f8e1a8;font-size:18px;font-weight:800;pointer-events:none}.portal-labels span:first-child{left:50%;top:3px;transform:translateX(-50%)}.portal-labels span:last-child{left:3px;top:50%;transform:translateY(-50%)}.board-dice-effect{position:absolute;inset:0;z-index:20;display:grid;place-items:center;align-content:center;gap:8px;background:rgba(22,27,33,.82);color:#fff}.board-die{display:grid;grid-template:repeat(3,12px)/repeat(3,12px);gap:3px;padding:12px;border-radius:10px;background:#f7f2e8;box-shadow:0 10px 25px rgba(0,0,0,.35);animation:dice-roll .7s cubic-bezier(.2,.8,.2,1)}.board-die i{width:8px;height:8px;border-radius:50%;background:transparent}.face-1 i:nth-child(5),.face-2 i:nth-child(1),.face-2 i:nth-child(9),.face-3 i:nth-child(1),.face-3 i:nth-child(5),.face-3 i:nth-child(9),.face-4 i:nth-child(1),.face-4 i:nth-child(3),.face-4 i:nth-child(7),.face-4 i:nth-child(9),.face-5 i:nth-child(1),.face-5 i:nth-child(3),.face-5 i:nth-child(5),.face-5 i:nth-child(7),.face-5 i:nth-child(9),.face-6 i:nth-child(1),.face-6 i:nth-child(3),.face-6 i:nth-child(4),.face-6 i:nth-child(6),.face-6 i:nth-child(7),.face-6 i:nth-child(9){background:#40372f}@keyframes tile-pop{0%{transform:scale(1)}50%{transform:scale(1.2)}100%{transform:scale(1)}}@keyframes tile-appear{0%{transform:scale(0);opacity:0}100%{transform:scale(1);opacity:1}}@keyframes dice-roll{0%{transform:translateY(-60px) rotate(-220deg) scale(.55);opacity:0}70%{transform:translateY(6px) rotate(16deg) scale(1.07)}100%{transform:none;opacity:1}}
.board-cell{position:absolute;float:none;margin:0}.board-cell,.board-tile{width:var(--cell-width,calc((100% - (var(--cols) + 1) * var(--gap))/var(--cols)));height:var(--cell-height,calc((100% - (var(--rows) + 1) * var(--gap))/var(--rows)))}.board-tile.no-transition{transition:none!important}.board-tile.hidden{opacity:0}.tournament-board.irregular{background:#a99d90}.tournament-board.irregular .board-cell{box-shadow:none}
.board-seals{position:absolute;inset:0;z-index:7;pointer-events:none}.board-seal{position:absolute;width:var(--cell-width,calc((100% - (var(--cols) + 1) * var(--gap))/var(--cols)));height:var(--cell-height,calc((100% - (var(--rows) + 1) * var(--gap))/var(--rows)));border-radius:8px;background:rgba(18,22,28,.43);box-shadow:inset 0 0 0 2px rgba(16,19,23,.88),inset 0 12px 18px rgba(0,0,0,.54),0 2px 8px rgba(0,0,0,.25)}.board-seal-icon{position:absolute;top:7px;right:7px;width:22px;height:22px;color:#f5ede2;filter:drop-shadow(0 1px 2px #000)}.board-seal-enter-active,.board-seal-leave-active{transition:opacity .2s ease,filter .2s ease}.board-seal-enter-from,.board-seal-leave-to{opacity:0;filter:blur(3px)}
.board-tile.pop{animation:tile-pop var(--board-pop-duration) ease backwards}
.board-tile.appear{animation:tile-appear var(--board-pop-duration) ease backwards}
</style>
