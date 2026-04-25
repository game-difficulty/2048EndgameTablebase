<template>
  <div class="board relative bg-board-bg rounded-xl aspect-square w-full max-w-[600px] mx-auto touch-none" style="container-type: size;">
    
    <!-- Grid Cells (Background) -->
    <div class="bg-grid">
      <div 
        v-for="i in 16" 
        :key="`bg-${i}`" 
        class="bg-cell pointer-events-auto"
        :style="getBackgroundCellStyle(i - 1)"
        @pointerdown.prevent="(e) => emit('cell-click', Math.floor((i-1)/4), (i-1)%4, e.button)"
        @contextmenu.prevent
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

  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue';

const emit = defineEmits(['cell-click']);

const props = defineProps({
  board: {
    type: Array,
    required: true,
    default: () => new Array(16).fill(0)
  },
  metadata: {
    type: Object,
    default: null
  },
  dis32k: {
    type: Boolean,
    default: false
  },
  isVariant: {
    type: Boolean,
    default: false
  }
});

let tileIdCounter = 0;
const activeTiles = ref([]);
let animTimeout = null;
let revealMergeTimeout = null;
let revealAppearTimeout = null;
const MERGE_GLOW_MIN_VALUE = 2048;
const MERGE_GLOW_STEPS = 5;

function isVariantWallValue(value) {
  return props.isVariant && Number(value) === 32768;
}

function shouldRenderAsActiveTile(value) {
  return Number(value) > 0 && !isVariantWallValue(value);
}

function hasMoveAnimationMetadata(metadata) {
  if (!metadata || typeof metadata !== 'object') return false;
  const appearTile = metadata.appear_tile;
  const hasAppearTile = !!appearTile
    && Number.isInteger(Number(appearTile.index))
    && Number(appearTile.index) >= 0
    && Number(appearTile.index) < 16
    && Number(appearTile.value) > 0;
  const { direction, slide_distances: slideDistances, pop_positions: popPositions } = metadata;
  const hasDirectionalAnimation = ['left', 'right', 'up', 'down'].includes(direction)
    && Array.isArray(slideDistances)
    && slideDistances.length === 16
    && Array.isArray(popPositions)
    && popPositions.length === 16;
  return hasDirectionalAnimation || hasAppearTile;
}

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

const syncToBoardRaw = () => {
    fastForwardAnimations(true);
    activeTiles.value = [];
    for(let i=0; i<16; i++) {
        if (shouldRenderAsActiveTile(props.board[i])) {
            activeTiles.value.push(withGlowDefaults({
                id: `tile-${tileIdCounter++}`,
                row: Math.floor(i / 4),
                col: i % 4,
                value: props.board[i],
                isDying: false,
                isMerged: false,
                isNew: false,
                isHidden: false,
                isInterrupting: false
            }));
        }
    }
};

watch(() => [props.board, props.isVariant], async ([newBoard]) => {
    if (!hasMoveAnimationMetadata(props.metadata)) {
        // Init or resync without animation
        clearAnimationTimers();
        syncToBoardRaw();
        return;
    }
    
    // Flush old animations logically
    clearAnimationTimers();
    fastForwardAnimations(true);
    activeTiles.value.forEach(tile => {
        tile.glowStepsRemaining = decayGlowSteps(tile);
    });

    // Force snap to DOM to prevent diagonal sliding
    activeTiles.value.forEach(t => t.isInterrupting = true);
    await nextTick();
    // Force browser reflow
    void document.body.offsetHeight;
    
    const { 
        direction = '', 
        slide_distances = [], 
        pop_positions = [], 
        appear_tile = null 
    } = props.metadata || {};
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
        if (pop_positions[newIndex] === 1) {
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
    
    revealMergeTimeout = setTimeout(() => {
        revealMergedTiles();
        revealMergeTimeout = null;
    }, 100);

    revealAppearTimeout = setTimeout(() => {
        revealAppearingTiles();
        revealAppearTimeout = null;
    }, 125);

    animTimeout = setTimeout(() => {
        fastForwardAnimations(false);
        animTimeout = null;
    }, 300);

}, { deep: true });

// Initial setup render
syncToBoardRaw();

const getTilePosStyle = (tile) => {
  return {
    '--col': tile.col,
    '--row': tile.row
  };
};

const getTileDisplayValue = (value) => {
  if (!value) return '';
  if (value === 32768 && (props.dis32k || isVariantWallValue(value))) return '';
  return value;
};

const getBackgroundCellStyle = (index) => (
  isVariantWallValue(props.board[index])
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
  /* Proportional: pad and gap as % of board container width via cqw */
  --padding: 2.5cqw;
  --grid-gap: 2.5cqw;
}

/* Background Grid aligns strictly to the padding offset */
.bg-grid {
  position: absolute;
  top: var(--padding);
  left: var(--padding);
  right: var(--padding);
  bottom: var(--padding);
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  grid-template-rows: repeat(4, 1fr);
  gap: var(--grid-gap);
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
  top: 0;
  left: 0;
  z-index: 10;
  
  /* The 100% inside this `calc()` refers to the width of `.board` */
  width: calc((100% - var(--padding) * 2 - var(--grid-gap) * 3) / 4);
  height: calc((100% - var(--padding) * 2 - var(--grid-gap) * 3) / 4);
  
  /* The 100% inside `translate` refers to `.tile`'s own width */
  transform: translate(
    calc(var(--padding) + var(--col) * (100% + var(--grid-gap))), 
    calc(var(--padding) + var(--row) * (100% + var(--grid-gap)))
  );
  
  /* Transition for when position properties update */
  transition: transform 0.1s ease-in-out;
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
  animation: appear 0.2s ease backwards;
}

.anim-merged {
  animation: pop 0.2s ease backwards;
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
