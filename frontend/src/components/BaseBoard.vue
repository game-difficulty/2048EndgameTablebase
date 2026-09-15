<template>
  <div
    ref="boardRef"
    class="board-stage relative aspect-square w-full max-w-[600px] mx-auto touch-none"
    :style="{ '--board-slide-duration': `${animationDuration / 3}ms`, '--board-pop-duration': `${animationDuration * 2 / 3}ms` }"
    @pointerdown.prevent="handleBoardPointerDown"
    @pointermove="handleBoardPointerMove"
    @pointerup="handleBoardPointerUp"
    @pointercancel="clearTouchGesture"
    @contextmenu.prevent
  >
    <div
      :class="['board absolute bg-board-bg rounded-xl', { 'board-compact': compact }]"
      :style="boardViewportStyle"
    >
      <!-- Grid Cells (Background) -->
      <div class="bg-grid">
        <div
          v-for="index in boardViewport.visibleIndices"
          :key="`bg-${index}`"
          class="bg-cell pointer-events-auto"
          :data-board-cell-index="index"
          :style="getBackgroundCellStyle(index)"
        ></div>
      </div>

      <!-- Active Tiles -->
      <div
        v-for="tile in activeTiles"
        :key="tile.id"
        class="tile z-10"
        :class="{'no-transition': tile.isInterrupting}"
        :style="getTilePosStyle(tile)"
      >
        <div
          class="tile-inner rounded-lg flex items-center justify-center font-bold"
          :class="{
            'anim-new': tile.isNew,
            'anim-merged': tile.isMerged && !tile.isHidden,
            'opacity-0': tile.isHidden
          }"
          :style="getTileInnerStyle(tile)"
        >
          <span class="tile-label" :style="getTileLabelStyle(tile)">
            {{ getTileDisplayValue(tile.value) }}
          </span>
        </div>
      </div>

      <slot name="overlay" />
    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch, nextTick, onUnmounted } from 'vue';

import {
  boardFrameRenderMode,
  cloneBoard,
} from './boardFrame.js';
import { boardSwipeDirection } from './boardPointerGesture.js';
import {
  boardViewportSignature,
  createBoardViewport,
  createBoardViewportLayout,
  logicalIndexToVisual,
} from '../utils/boardViewport.js';

const emit = defineEmits(['cell-click', 'swipe']);

const props = defineProps({
  animationDuration: { type: Number, default: 300 },
  frame: {
    type: Object,
    required: true
  },
  dis32k: {
    type: Boolean,
    default: false
  },
  compact: {
    type: Boolean,
    default: false
  },
  isVariant: {
    type: Boolean,
    default: false
  }
});

let tileIdCounter = 0;
const boardRef = ref(null);
const activeTiles = ref([]);
const boardViewport = computed(() => createBoardViewport(props.frame?.toBoard, props.isVariant));
const viewportLayout = computed(() => createBoardViewportLayout(boardViewport.value));
const viewportSignature = computed(() => boardViewportSignature(boardViewport.value));
const boardViewportStyle = computed(() => {
  const layout = viewportLayout.value;
  return {
    width: `${layout.widthPercent}%`,
    height: `${layout.heightPercent}%`,
    left: `${layout.leftPercent}%`,
    top: `${layout.topPercent}%`,
    '--visible-rows': boardViewport.value.rows,
    '--visible-cols': boardViewport.value.cols,
    '--board-padding-x': `${layout.paddingXPercent}%`,
    '--board-padding-y': `${layout.paddingYPercent}%`,
    '--grid-gap-x': `${layout.gapXPercent}%`,
    '--grid-gap-y': `${layout.gapYPercent}%`,
    '--tile-width': `${layout.tileWidthPercent}%`,
    '--tile-height': `${layout.tileHeightPercent}%`,
  };
});
let animTimeout = null;
let revealMergeTimeout = null;
let revealAppearTimeout = null;
let animationEpoch = 0;
let lastConsumedFrameRevision = null;
let settledBoard = cloneBoard(props.frame?.toBoard);
const MERGE_GLOW_MIN_VALUE = 2048;
const MERGE_GLOW_STEPS = 5;
let touchGesture = null;

function isVariantWallValue(value) {
  return props.isVariant && Number(value) === 32768;
}

function isVariantNonMergingValue(value) {
  return props.isVariant && Number(value) === 16384;
}

function shouldRenderAsActiveTile(value) {
  return Number(value) > 0 && !isVariantWallValue(value);
}

const clearTouchGesture = () => {
  if (touchGesture?.pointerId != null && boardRef.value?.hasPointerCapture?.(touchGesture.pointerId)) {
    try {
      boardRef.value.releasePointerCapture(touchGesture.pointerId);
    } catch {
      // Ignore stale captures from interrupted gestures.
    }
  }
  touchGesture = null;
};

const eventCellIndex = (event) => {
  const cell = event.target?.closest?.('[data-board-cell-index]');
  if (!cell || !boardRef.value?.contains(cell)) return null;
  const index = Number(cell.dataset.boardCellIndex);
  return Number.isInteger(index) && index >= 0 && index < 16 ? index : null;
};

const handleBoardPointerDown = (event) => {
  const index = eventCellIndex(event);
  if (event.pointerType === 'mouse') {
    if (index != null) {
      emit('cell-click', Math.floor(index / 4), index % 4, event.button);
    }
    return;
  }

  touchGesture = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    lastX: event.clientX,
    lastY: event.clientY,
    cellIndex: index,
    button: event.button,
  };

  if (boardRef.value?.setPointerCapture) {
    try {
      boardRef.value.setPointerCapture(event.pointerId);
    } catch {
      // Pointer capture is best-effort for touch drags.
    }
  }
};

const handleBoardPointerMove = (event) => {
  if (!touchGesture || event.pointerId !== touchGesture.pointerId) {
    return;
  }
  touchGesture.lastX = event.clientX;
  touchGesture.lastY = event.clientY;
};

const handleBoardPointerUp = (event) => {
  if (!touchGesture || event.pointerId !== touchGesture.pointerId) {
    return;
  }

  touchGesture.lastX = event.clientX;
  touchGesture.lastY = event.clientY;
  const dx = touchGesture.lastX - touchGesture.startX;
  const dy = touchGesture.lastY - touchGesture.startY;
  const boardBounds = boardRef.value?.getBoundingClientRect?.();
  const displaySize = Math.min(Number(boardBounds?.width) || 0, Number(boardBounds?.height) || 0);
  const direction = boardSwipeDirection(dx, dy, displaySize);
  const { cellIndex, button } = touchGesture;
  clearTouchGesture();

  if (direction) {
    emit('swipe', direction);
    return;
  }

  if (cellIndex != null) {
    emit('cell-click', Math.floor(cellIndex / 4), cellIndex % 4, button);
  }
};

const decayGlowSteps = (tile) => {
    if (!tile.glowStepsRemaining) return 0;
    return Math.max(0, tile.glowStepsRemaining - 1);
};

const withGlowDefaults = (tile, glowStepsRemaining = 0) => ({
    glowStepsRemaining,
    ...tile,
});

// Clean up animations and flush visual state
const fastForwardAnimations = (isInterrupt = false) => {
    // 1. Remove dying tiles (those merged into others)
    activeTiles.value = activeTiles.value.filter(t => !t.isDying);
    
    // 2. Unhide merged tiles and clear all animation flags
    activeTiles.value.forEach(t => {
        t.isNew = false;
        t.isHidden = false;
        t.isMerged = false; // Always clear to prevent replay on v-show toggle
    });
};

const clearAnimationTimers = () => {
    if (animTimeout) {
        clearTimeout(animTimeout);
        animTimeout = null;
    }
    if (revealMergeTimeout) {
        clearTimeout(revealMergeTimeout);
        revealMergeTimeout = null;
    }
    if (revealAppearTimeout) {
        clearTimeout(revealAppearTimeout);
        revealAppearTimeout = null;
    }
};

const revealMergedTiles = () => {
    activeTiles.value = activeTiles.value.map(tile => {
        if (tile.isDying) {
            return { ...tile, isHidden: true };
        }
        if (tile.isMerged && tile.isHidden) {
            return { ...tile, isHidden: false };
        }
        return tile;
    });
};

const revealAppearingTiles = () => {
    activeTiles.value = activeTiles.value.map(tile => {
        if (tile.isNew && tile.isHidden) {
            return { ...tile, isHidden: false };
        }
        return tile;
    });
};

const syncToBoardRaw = (sourceBoard = props.frame?.toBoard) => {
    fastForwardAnimations(true);
    const normalizedBoard = cloneBoard(sourceBoard);
    const nextTiles = [];
    for (const i of boardViewport.value.visibleIndices) {
        if (shouldRenderAsActiveTile(normalizedBoard[i])) {
            nextTiles.push(withGlowDefaults({
                // Snapshot updates use cell-stable keys so undo/seek does not
                // destroy and recreate every visible tile.
                id: `snapshot-${i}`,
                row: Math.floor(i / 4),
                col: i % 4,
                value: normalizedBoard[i],
                isDying: false,
                isMerged: false,
                isNew: false,
                isHidden: false,
                isInterrupting: true
            }));
        }
    }
    activeTiles.value = nextTiles;
    settledBoard = normalizedBoard;
};

watch(
  () => [props.frame?.revision ?? null, props.isVariant, viewportSignature.value],
  async ([revision, variant, viewportKey], previous = []) => {
    const frame = props.frame;
    if (!frame) return;
    const frameRevision = String(revision ?? '');
    const variantChanged = previous.length > 0 && variant !== previous[1];
    const viewportChanged = previous.length > 0 && viewportKey !== previous[2];
    if (!variantChanged && !viewportChanged && frameRevision === lastConsumedFrameRevision) return;
    lastConsumedFrameRevision = frameRevision;

    const epoch = ++animationEpoch;
    clearAnimationTimers();
    fastForwardAnimations(true);

    const newBoard = cloneBoard(frame.toBoard);
    if (variantChanged || viewportChanged || boardFrameRenderMode(settledBoard, frame) !== 'animate') {
        syncToBoardRaw(newBoard);
        return;
    }

    const animationMetadata = frame.metadata;
    activeTiles.value.forEach(tile => {
        tile.glowStepsRemaining = decayGlowSteps(tile);
    });

    // Force snap to DOM to prevent diagonal sliding
    activeTiles.value.forEach(t => t.isInterrupting = true);
    await nextTick();
    if (epoch !== animationEpoch) return;
    // Force browser reflow
    void document.body.offsetHeight;
    
    const { 
        direction = '', 
        slide_distances = [], 
        pop_positions = [], 
        appear_tile = null 
    } = animationMetadata || {};
    const vectors = {
        'left': { x: -1, y: 0 },
        'right': { x: 1, y: 0 },
        'up': { x: 0, y: -1 },
        'down': { x: 0, y: 1 }
    };
    
    const v = vectors[direction] || {x: 0, y:0};
    let newActive = [];
    
    // Apply logic changes to existing DOM tiles
    activeTiles.value.forEach(tile => {
        const oldIndex = tile.row * 4 + tile.col;
        const dist = slide_distances[oldIndex];
        
        let tx = tile.col;
        let ty = tile.row;
        if (dist > 0) {
            tx += v.x * dist;
            ty += v.y * dist;
            // Native reactivity triggers CSS translate wrapper shift
            tile.col = tx;
            tile.row = ty; 
        }      
        const newIndex = ty * 4 + tx;
        const shouldMergeTile = pop_positions[newIndex] === 1 && !isVariantNonMergingValue(tile.value);
        if (shouldMergeTile) {
            tile.isDying = true; // Mark old tile to eventually die
            
            // Generate the ultimate merged tile hidden
            if (shouldRenderAsActiveTile(newBoard[newIndex]) && !newActive.find(t => t.col === tx && t.row === ty && t.isHidden)) {
               newActive.push(withGlowDefaults({
                   id: `tile-${tileIdCounter++}`,
                   row: ty,
                   col: tx,
                   value: newBoard[newIndex],
                   isNew: false,
                   isMerged: true,
                   isDying: false,
                   isHidden: true, // Hide it while the original pieces slide
                   isInterrupting: false
               }, newBoard[newIndex] >= MERGE_GLOW_MIN_VALUE ? MERGE_GLOW_STEPS : 0));
            }
        }
        tile.isInterrupting = false; // Restore transition for sliding
        newActive.push(tile);
    });
    
    // Push the newest spawned tile
    if (appear_tile && shouldRenderAsActiveTile(appear_tile.value)) {
        newActive.push(withGlowDefaults({
            id: `tile-${tileIdCounter++}`,
            row: Math.floor(appear_tile.index / 4),
            col: appear_tile.index % 4,
            value: appear_tile.value,
            isNew: true,
            isMerged: false,
            isDying: false,
            isHidden: true,
            isInterrupting: false
        }));
    }
    
    activeTiles.value = newActive;
    settledBoard = newBoard;
    
    revealMergeTimeout = setTimeout(() => {
        if (epoch !== animationEpoch) return;
        revealMergedTiles();
        revealMergeTimeout = null;
    }, props.animationDuration / 3);

    revealAppearTimeout = setTimeout(() => {
        if (epoch !== animationEpoch) return;
        revealAppearingTiles();
        revealAppearTimeout = null;
    }, props.animationDuration * 5 / 12);

    animTimeout = setTimeout(() => {
        if (epoch !== animationEpoch) return;
        fastForwardAnimations(false);
        animTimeout = null;
    }, props.animationDuration);
  },
);

// Initial setup render
syncToBoardRaw();

onUnmounted(() => {
    animationEpoch += 1;
    clearAnimationTimers();
});

const getTilePosStyle = (tile) => {
  const visualPosition = logicalIndexToVisual(tile.row * 4 + tile.col, boardViewport.value);
  if (!visualPosition) {
    return { display: 'none' };
  }
  const layout = viewportLayout.value;
  return {
    left: `${layout.paddingXPercent + visualPosition.col * (layout.tileWidthPercent + layout.gapXPercent)}%`,
    top: `${layout.paddingYPercent + visualPosition.row * (layout.tileHeightPercent + layout.gapYPercent)}%`,
  };
};

const getTileDisplayValue = (value) => {
  if (!value) return '';
  if (value === 32768 && (props.dis32k || isVariantWallValue(value))) return '';
  return value;
};

const getBackgroundCellStyle = (index) => (
  isVariantWallValue(props.frame?.toBoard?.[index])
    ? { backgroundColor: 'var(--color-board-bg)' }
    : null
);

const getTileInnerStyle = (tile) => {
  if (isVariantWallValue(tile.value)) {
    return {
      backgroundColor: 'var(--color-board-bg)',
      color: 'transparent',
      boxShadow: 'none',
    };
  }

  const glowRatio = tile.glowStepsRemaining > 0
    ? tile.glowStepsRemaining / MERGE_GLOW_STEPS
    : 0;
  const glowAlpha = (0.18 + glowRatio * 0.28).toFixed(3);
  const glowSpread = `${8 + glowRatio * 12}px`;
  const glowOuter = `${16 + glowRatio * 20}px`;

  return {
    backgroundColor: `var(--color-tile-${tile.value})`,
    color: `var(--color-text-${tile.value})`,
    boxShadow: glowRatio > 0
      ? `0 0 ${glowSpread} rgba(255, 214, 102, ${glowAlpha}), 0 0 ${glowOuter} rgba(255, 214, 102, ${(glowRatio * 0.22).toFixed(3)}), inset 0 0 0 1px rgba(255,255,255,${(0.08 + glowRatio * 0.12).toFixed(3)})`
      : 'none'
  };
};

const getTileLabelStyle = (tile) => {
  const len = String(tile.value).length;
  let fontSize = '2.5rem';
  let textOffset = '0.015em';

  if (len > 4) {
    fontSize = '1.5rem';
    textOffset = '0.05em';
  } else if (len > 3) {
    fontSize = '2rem';
    textOffset = '0.04em';
  } else if (len === 3) {
    textOffset = '0.03em';
  } else if (len === 2) {
    textOffset = '0.02em';
  }

  return {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: `calc(${fontSize} * var(--tile-font-scale, 1))`,
    lineHeight: 1,
    transform: `translateY(${textOffset})`,
  };
};
</script>

<style scoped>
.board {
  --visible-rows: 4;
  --visible-cols: 4;
  --board-padding-x: 2.5%;
  --board-padding-y: 2.5%;
  --grid-gap-x: 2.5%;
  --grid-gap-y: 2.5%;
  --tile-width: 21.875%;
  --tile-height: 21.875%;
}

.board-compact {
  --tile-font-scale: 0.45;
}

/* Background Grid aligns strictly to the padding offset */
.bg-grid {
  position: absolute;
  top: var(--board-padding-y);
  left: var(--board-padding-x);
  right: var(--board-padding-x);
  bottom: var(--board-padding-y);
  display: grid;
  grid-template-columns: repeat(var(--visible-cols), 1fr);
  grid-template-rows: repeat(var(--visible-rows), 1fr);
  column-gap: var(--grid-gap-x);
  row-gap: var(--grid-gap-y);
  z-index: 0;
}

.bg-cell {
  background-color: var(--color-empty);
  border-radius: 0.5rem; /* rounded-lg */
  width: 100%;
  height: 100%;
}

/* Foreground Tiles use explicit math off the board boundary */
.tile {
  pointer-events: none;
  position: absolute;
  z-index: 10;
  width: var(--tile-width);
  height: var(--tile-height);
  transition: top var(--board-slide-duration, 100ms) ease-in-out, left var(--board-slide-duration, 100ms) ease-in-out;
}

.no-transition {
  transition: none !important;
}

/* Inner block */
.tile-inner {
  width: 100%;
  height: 100%;
  line-height: 1;
  transition: background-color 0.15s ease, color 0.15s ease, box-shadow 0.2s ease, opacity 0s;
}

.tile-label {
  min-width: 0;
  max-width: 100%;
  text-align: center;
  line-height: 1;
  transition: transform 0.1s ease;
}

.anim-new {
  animation: appear var(--board-pop-duration, 200ms) ease backwards;
}

.anim-merged {
  animation: pop var(--board-pop-duration, 200ms) ease backwards;
}

@keyframes appear {
  0% { transform: scale(0); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}

@keyframes pop {
  0% { transform: scale(1); }
  50% { transform: scale(1.2); }
  100% { transform: scale(1); }
}
</style>
