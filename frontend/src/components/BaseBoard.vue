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
import { computed, ref } from 'vue';
import { useBoardAnimation } from './useBoardAnimation.js';

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

const boardRef = ref(null);
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
const { activeTiles } = useBoardAnimation(props, boardViewport, viewportSignature);
const MERGE_GLOW_STEPS = 5;
let touchGesture = null;

function isVariantWallValue(value) {
  return props.isVariant && Number(value) === 32768;
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
  const smallTileScale = tile.value >= 2 && tile.value <= 64 ? 1.2 : 1;
  let fontSize = 'var(--tile-label-small, 2.5rem)';
  let textOffset = '0.015em';

  if (len > 4) {
    fontSize = 'var(--tile-label-large, 1.5rem)';
    textOffset = '0.05em';
  } else if (len > 3) {
    fontSize = 'var(--tile-label-medium, 2rem)';
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
    fontSize: `calc(${fontSize} * var(--tile-font-scale, 1) * ${smallTileScale})`,
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
