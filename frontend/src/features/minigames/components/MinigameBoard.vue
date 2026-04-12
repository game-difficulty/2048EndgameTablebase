<template>
  <div
    ref="boardRef"
    class="board relative bg-board-bg rounded-xl w-full max-w-[600px] mx-auto touch-none overflow-hidden"
    :class="{ 'has-custom-cursor': Boolean(props.interaction?.active) }"
    :style="boardShellStyle"
    @pointermove="handlePointerMove"
    @pointerleave="handlePointerLeave"
  >
    <div class="minigame-board-grid" :style="boardGridStyle">
      <button
        v-for="cell in cells"
        :key="cell.index"
        type="button"
        :disabled="cell.blocked"
        :class="['minigame-cell', { 'is-hole-cell': cell.isHole }]"
        :style="getBackgroundCellStyle(cell)"
        @click="handleGridCellClick(cell)"
      >
        <template v-if="!hasAnimatedTile(cell.index) && !isEffectHidden(cell.index)">
          <div v-if="cell.coverSprites?.length" class="minigame-cover-stack">
            <img
              v-for="(sprite, spriteIndex) in cell.coverSprites"
              :key="`${cell.index}-cover-${spriteIndex}`"
              :src="getCoverSpriteSrc(sprite)"
              :class="getCoverSpriteClass(sprite)"
              alt=""
              draggable="false"
            />
          </div>
          <span v-if="cell.overlay" class="minigame-cell-overlay">{{ getOverlayText(cell.overlay) }}</span>
          <span
            v-if="showStaticCellLabel(cell)"
            :class="['minigame-cell-label', getTileLabelClass(cell)]"
          >{{ cell.displayText }}</span>
        </template>
      </button>
    </div>

    <div class="minigame-active-layer">
      <div
        v-for="tile in activeTiles"
        :key="tile.id"
        class="minigame-tile"
        :class="{ 'no-transition': tile.isInterrupting }"
        :style="getTilePosStyle(tile)"
      >
        <div
          class="minigame-tile-inner"
          :class="{
            'anim-new': tile.isNew,
            'anim-merged': tile.isPopActive && !tile.isHidden,
            'opacity-0': tile.isHidden,
          }"
          :style="getTileInnerStyle(tile)"
        >
          <div v-if="tile.coverSprites?.length" class="minigame-cover-stack">
            <img
              v-for="(sprite, spriteIndex) in tile.coverSprites"
              :key="`${tile.id}-cover-${spriteIndex}`"
              :src="getCoverSpriteSrc(sprite)"
              :class="getCoverSpriteClass(sprite)"
              alt=""
              draggable="false"
            />
          </div>
          <span v-if="tile.overlay" class="minigame-cell-overlay">{{ getOverlayText(tile.overlay) }}</span>
          <span :class="['minigame-cell-label', getTileLabelClass(tile)]" :style="getTileLabelStyle(tile)">
            {{ tile.displayText }}
          </span>
        </div>
      </div>
    </div>

    <div class="minigame-small-label-layer">
      <div
        v-for="cell in labeledCells"
        :key="`small-label-${cell.index}`"
        class="minigame-small-label-slot"
        :style="getCellSlotStyle(cell)"
      >
        <span class="minigame-cell-small">{{ cell.smallLabel }}</span>
      </div>
    </div>

    <div class="minigame-effects-layer">
      <div
        v-for="effect in specialEffects"
        :key="effect.id"
        class="minigame-effect"
        :style="getEffectWrapperStyle(effect)"
      >
        <div
          :class="[
            effect.type === 'explosion'
              ? 'effect-explosion'
              : effect.type === 'airraid_fire_drop'
                ? 'effect-airraid-fire-drop'
                : effect.type === 'airraid_explosion'
                  ? 'effect-airraid-explosion'
                  : effect.type === 'object_slide'
                    ? 'effect-object-slide'
                    : effect.type === 'giftbox_burst'
                      ? 'effect-giftbox-burst'
                      : effect.type === 'factorization_burst'
                        ? 'effect-factorization-burst'
                        : effect.type === 'ice_stage_reveal'
                          ? 'effect-ice-stage-reveal'
                          : effect.type === 'grab'
                            ? 'minigame-tile-inner effect-grab'
                            : effect.type === 'glove_move'
                              ? 'minigame-tile-inner effect-glove-move'
                              : effect.type === 'column_swap_move'
                                ? 'minigame-tile-inner effect-column-swap-move'
                                : effect.type === 'ring_rotate_move'
                                  ? 'minigame-tile-inner effect-ring-rotate-move'
                                  : effect.type === 'gravity_fall_move'
                                    ? 'minigame-tile-inner effect-gravity-fall-move'
                                    : effect.type === 'twist_move'
                                      ? 'minigame-tile-inner effect-twist-move'
                                      : 'effect-twist',
          ]"
          :style="getEffectInnerStyle(effect)"
        >
          <img
            v-if="shouldRenderEffectImage(effect)"
            :src="getEffectImageSrc(effect)"
            class="effect-image"
            :class="getEffectImageClass(effect)"
            alt=""
            draggable="false"
          />
          <template v-if="shouldRenderEffectTile(effect)">
            <img
              v-if="effect.type === 'grab' || effect.type === 'glove_move'"
              :src="getEffectImageSrc(effect)"
              class="effect-image effect-glove-image"
              alt=""
              draggable="false"
            />
            <span :class="['minigame-cell-label', getTileLabelClass(effect)]" :style="getTileLabelStyle(effect)">
              {{ effect.labelText || effect.displayText }}
            </span>
          </template>
        </div>
      </div>
    </div>

    <div v-if="isTwistMode" class="minigame-hotspots-layer">
      <button
        v-for="hotspot in twistHotspots"
        :key="`twist-${hotspot.index}`"
        type="button"
        class="twist-hotspot"
        :style="getTwistHotspotStyle(hotspot)"
        @click.stop="$emit('cell-click', hotspot)"
      ></button>
    </div>

    <div
      v-if="cursorPreviewVisible && cursorPreviewSrc"
      class="cursor-preview"
      :style="cursorPreviewStyle"
    >
      <img :src="cursorPreviewSrc" alt="" draggable="false" class="cursor-preview-image" />
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue';

import {
  buildBoardCells,
  exponentToTileValue,
  formatBoardTileNumber,
  getBoardCellStyle,
} from '../model/minigameMappers';

const props = defineProps({
  board: {
    type: Array,
    default: () => new Array(16).fill(0),
  },
  shape: {
    type: Object,
    default: () => ({ rows: 4, cols: 4 }),
  },
  view: {
    type: Object,
    default: () => ({}),
  },
  metadata: {
    type: Object,
    default: () => ({}),
  },
  interaction: {
    type: Object,
    default: () => ({
      active: false,
      validTargets: [],
      selectedIndices: [],
      mode: null,
      phase: null,
    }),
  },
});

const emit = defineEmits(['cell-click']);

const boardRef = ref(null);
const activeTiles = ref([]);
const settledTiles = ref([]);
const specialEffects = ref([]);
const effectHiddenIndices = ref(new Set());
const cursorPreview = ref({ x: 0, y: 0, visible: false });
let tileIdCounter = 0;
let revealTimeout = null;
let moveCleanupTimeout = null;
let effectsCleanupTimeout = null;
let followUpTimeout = null;
const assetBase = 'http://127.0.0.1:8000/minigames-assets';

const rows = computed(() => Number(props.shape?.rows || 4));
const cols = computed(() => Number(props.shape?.cols || 4));
const cellCount = computed(() => rows.value * cols.value);
const cells = computed(() => buildBoardCells(props.board, props.shape, props.view));
const cellMap = computed(() => new Map(cells.value.map((cell) => [cell.index, cell])));
const labeledCells = computed(() =>
  cells.value.filter((cell) => Boolean(cell?.smallLabel) && !cell.blocked)
);
const activeIndexSet = computed(() => new Set(activeTiles.value.map((tile) => tile.index)));
const interactionValidTargets = computed(
  () => new Set((props.interaction?.validTargets || []).map((index) => Number(index)))
);
const interactionSelected = computed(
  () => new Set((props.interaction?.selectedIndices || []).map((index) => Number(index)))
);
const isTwistMode = computed(
  () => Boolean(props.interaction?.active && props.interaction?.mode === 'twist')
);
const twistHotspots = computed(() =>
  (props.interaction?.validTargets || []).map((index) => {
    const numericIndex = Number(index);
    return {
      index: numericIndex,
      row: Math.floor(numericIndex / cols.value),
      col: numericIndex % cols.value,
    };
  })
);
const cursorPreviewSrc = computed(() => {
  if (!props.interaction?.active) return '';
  if (props.interaction.mode === 'bomb') return `${assetBase}/bomb2.png`;
  if (props.interaction.mode === 'glove') return `${assetBase}/glove.png`;
  if (props.interaction.mode === 'twist') return `${assetBase}/twist.png`;
  return '';
});
const cursorPreviewVisible = computed(
  () => Boolean(props.interaction?.active && cursorPreview.value.visible && cursorPreviewSrc.value)
);
const cursorPreviewStyle = computed(() => ({
  left: `${cursorPreview.value.x}px`,
  top: `${cursorPreview.value.y}px`,
}));
const maxShapeDimension = computed(() => Math.max(rows.value, cols.value, 4));
const shapeDensityScale = computed(() => Math.min(1, 5.5 / maxShapeDimension.value));
const shapeTextScale = computed(() => Math.min(1, 6 / maxShapeDimension.value));
const boardShellStyle = computed(() => ({
  containerType: 'size',
  aspectRatio: `${cols.value} / ${rows.value}`,
  '--shape-density-scale': shapeDensityScale.value.toFixed(4),
  '--shape-text-scale': shapeTextScale.value.toFixed(4),
}));

const boardGridStyle = computed(() => ({
  gridTemplateColumns: `repeat(${cols.value}, minmax(0, 1fr))`,
  gridTemplateRows: `repeat(${rows.value}, minmax(0, 1fr))`,
}));

const clearAnimationTimers = () => {
  if (revealTimeout) {
    window.clearTimeout(revealTimeout);
    revealTimeout = null;
  }
  if (moveCleanupTimeout) {
    window.clearTimeout(moveCleanupTimeout);
    moveCleanupTimeout = null;
  }
  if (effectsCleanupTimeout) {
    window.clearTimeout(effectsCleanupTimeout);
    effectsCleanupTimeout = null;
  }
  if (followUpTimeout) {
    window.clearTimeout(followUpTimeout);
    followUpTimeout = null;
  }
};

const clearSpecialEffects = () => {
  specialEffects.value = [];
  effectHiddenIndices.value = new Set();
};

const getOverlayText = (overlay) => {
  if (!overlay) return '';
  if (typeof overlay === 'object') return String(overlay.text || '');
  return String(overlay);
};

const handlePointerMove = (event) => {
  if (!props.interaction?.active || !boardRef.value) {
    cursorPreview.value.visible = false;
    return;
  }
  const rect = boardRef.value.getBoundingClientRect();
  cursorPreview.value = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
    visible: true,
  };
};

const handlePointerLeave = () => {
  cursorPreview.value.visible = false;
};

const handleGridCellClick = (cell) => {
  if (props.interaction?.mode === 'twist') return;
  emit('cell-click', cell);
};

const getEffectImageSrc = (effect) => {
  if (effect?.sprite) return getCoverSpriteSrc(effect.sprite);
  if (effect.type === 'explosion') return `${assetBase}/explode.png`;
  if (effect.type === 'airraid_fire_drop') return `${assetBase}/fire.png`;
  if (effect.type === 'airraid_explosion') return `${assetBase}/explode.png`;
  if (effect.type === 'giftbox_burst') return `${assetBase}/hole.png`;
  if (effect.type === 'factorization_burst') return `${assetBase}/hole2.png`;
  if (effect.type === 'grab' || effect.type === 'glove_move') return `${assetBase}/glove.png`;
  if (effect.type === 'twist') return `${assetBase}/twist.png`;
  return '';
};

const shouldRenderEffectImage = (effect) =>
  [
    'explosion',
    'twist',
    'airraid_fire_drop',
    'airraid_explosion',
    'object_slide',
    'giftbox_burst',
    'factorization_burst',
    'ice_stage_reveal',
  ].includes(String(effect?.type || ''));

const shouldRenderEffectTile = (effect) =>
  [
    'grab',
    'glove_move',
    'twist_move',
    'column_swap_move',
    'ring_rotate_move',
  ].includes(String(effect?.type || ''));

const getEffectImageClass = (effect) => {
  const type = String(effect?.type || '');
  const spriteCoverClasses =
    (type === 'ice_stage_reveal' || type === 'object_slide') && effect?.sprite
      ? getCoverSpriteClass(effect.sprite)
      : [];
  return [
    'effect-image',
    type === 'explosion' || type === 'airraid_explosion' ? 'effect-explosion-image' : '',
    type === 'twist' ? 'effect-twist-image' : '',
    type === 'airraid_fire_drop' ? 'effect-fire-image' : '',
    type === 'giftbox_burst' ? 'effect-giftbox-image' : '',
    type === 'factorization_burst' ? 'effect-factorization-image' : '',
    type === 'ice_stage_reveal' ? 'effect-ice-reveal-image' : '',
    type === 'object_slide' ? 'effect-object-image' : '',
    ...spriteCoverClasses,
  ].filter(Boolean);
};

const getCoverSpriteSrc = (sprite) => {
  const normalized = String(sprite || '').toLowerCase();
  if (normalized === 'bomb.png') return `${assetBase}/bomb2.png`;
  if (normalized === 'giftbox.png') return 'http://127.0.0.1:8000/shared-assets/giftbox2.png';
  return `${assetBase}/${sprite}`;
};

const getCoverSpriteClass = (sprite) => {
  const normalized = String(sprite || '').toLowerCase();
  return [
    'minigame-cover-sprite',
    normalized === 'portal.png' ? 'cover-sprite-portal' : '',
    normalized === 'target.png' ? 'cover-sprite-target' : '',
    normalized === 'bomb.png' ? 'cover-sprite-bomb' : '',
    normalized === 'giftbox.png' ? 'cover-sprite-giftbox' : '',
    normalized === 'tilebg.png' ? 'cover-sprite-tilebg' : '',
    normalized === 'ice_overlay.png' ? 'cover-sprite-ice-overlay' : '',
    normalized === 'icetrap0.png' ? 'cover-sprite-icetrap0' : '',
    normalized === 'icetrap.png' ? 'cover-sprite-icetrap' : '',
    normalized === 'crystal1.png' ? 'cover-sprite-crystal1' : '',
    normalized === 'crystal2.png' ? 'cover-sprite-crystal2' : '',
    normalized === 'crystal3.png' ? 'cover-sprite-crystal3' : '',
    normalized === 'fire.png' ? 'cover-sprite-fire' : '',
  ].filter(Boolean);
};

const getTileLabelClass = (tile) => {
  const variant = tile?.variant && typeof tile.variant === 'object' ? tile.variant : null;
  return {
    'label-mystery-hidden': variant?.kind === 'mystery-hidden' || variant?.kind === 'tilebg',
    'label-frozen': variant?.kind === 'frozen',
  };
};

const isEffectHidden = (index) => effectHiddenIndices.value.has(index);

const isRenderableTile = (cell) => Boolean(cell && !cell.blocked && cell.rawValue !== 0);

const createTileFromCell = (cell, overrides = {}) => {
  const style = getBoardCellStyle(cell);
  return {
    id: `minigame-tile-${tileIdCounter++}`,
    index: cell.index,
    row: cell.row,
    col: cell.col,
    rawValue: cell.rawValue,
    tileValue: cell.tileValue,
    displayText: cell.displayText,
    smallLabel: cell.smallLabel,
    overlay: cell.overlay,
    variant: cell.variant,
    coverSprites: Array.isArray(cell.coverSprites) ? [...cell.coverSprites] : [],
    background: style.background,
    color: style.color,
    borderColor: style.borderColor,
    boxShadow: style.boxShadow,
    isNew: false,
    isMerged: false,
    isPopActive: false,
    isDying: false,
    isHidden: false,
    isInterrupting: false,
    ...overrides,
  };
};

const createSyntheticTile = (index, rawValue, overrides = {}) => {
  const normalizedRawValue = Number(rawValue || 0);
  const tileValue = exponentToTileValue(normalizedRawValue);
  const safeValue = Math.min(tileValue || 0, 131072) || 2;
  return {
    id: `minigame-effect-${tileIdCounter++}`,
    index,
    row: Math.floor(index / cols.value),
    col: index % cols.value,
    rawValue: normalizedRawValue,
    tileValue,
    displayText: normalizedRawValue > 0 ? formatBoardTileNumber(tileValue) : '',
    smallLabel: '',
    overlay: null,
    variant: null,
    background: `var(--color-tile-${safeValue})`,
    color: `var(--color-text-${safeValue})`,
    borderColor: 'rgba(255,255,255,0.08)',
    boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.08)',
    isNew: false,
    isMerged: false,
    isPopActive: false,
    isDying: false,
    isHidden: false,
    isInterrupting: false,
    ...overrides,
  };
};

const createMergedAnimationTile = (targetIndex, sourceTile) => {
  const finalCell = cellMap.value.get(targetIndex);
  if (finalCell && isRenderableTile(finalCell)) {
    return createTileFromCell(finalCell, {
      row: Math.floor(targetIndex / cols.value),
      col: targetIndex % cols.value,
      index: targetIndex,
      isMerged: true,
      isPopActive: false,
      isHidden: true,
    });
  }

  if (Number(sourceTile?.rawValue || 0) > 0) {
    return createSyntheticTile(targetIndex, Number(sourceTile.rawValue) + 1, {
      row: Math.floor(targetIndex / cols.value),
      col: targetIndex % cols.value,
      index: targetIndex,
      isMerged: true,
      isPopActive: false,
      isHidden: true,
    });
  }

  return null;
};

const createObjectEffect = (effect) => ({
  id: `minigame-effect-${tileIdCounter++}`,
  type: String(effect.type || ''),
  index: Number(effect.fromIndex ?? effect.index ?? 0),
  row: Math.floor(Number(effect.fromIndex ?? effect.index ?? 0) / cols.value),
  col: Number(effect.fromIndex ?? effect.index ?? 0) % cols.value,
  targetRow: Math.floor(Number(effect.toIndex ?? effect.index ?? 0) / cols.value),
  targetCol: Number(effect.toIndex ?? effect.index ?? 0) % cols.value,
  sprite: String(effect.sprite || ''),
  labelText: String(effect.labelText || ''),
  displayText: String(effect.labelText || ''),
  variant:
    effect.sprite === 'tilebg.png'
      ? { kind: 'tilebg' }
      : effect.sprite === 'bomb.png' || effect.sprite === 'giftbox.png'
        ? { kind: 'special-object' }
        : null,
  background: 'transparent',
  color: '#fff',
  borderColor: 'transparent',
  boxShadow: 'none',
  durationMs: Number(effect.durationMs || effect.animDurationMs || 420),
  animDurationMs: Number(effect.animDurationMs || effect.durationMs || 420),
  delayMs: Number(effect.delayMs || 0),
  fadeOutAtEnd: Boolean(effect.fadeOutAtEnd),
});

const hasAnimatedTile = (index) => activeIndexSet.value.has(index);

const showStaticCellLabel = (cell) =>
  Boolean(!cell.blocked && cell.displayText && !hasAnimatedTile(cell.index));

const getBackgroundCellStyle = (cell) => {
  const isValidTarget =
    props.interaction?.mode !== 'twist' && interactionValidTargets.value.has(cell.index);
  const isSelected = interactionSelected.value.has(cell.index);
  if (cell.blocked) {
    const blockedStyle = getBoardCellStyle(cell);
    return {
      ...blockedStyle,
      background: blockedStyle.background,
      color: blockedStyle.color,
      borderColor: blockedStyle.borderColor,
      opacity: cell.isHole ? '0' : '0.75',
    };
  }
  const baseStyle = {
    background:
      cell?.variant?.kind === 'portal' ||
      cell?.variant?.kind === 'tilebg' ||
      cell?.variant?.kind === 'mystery-hidden'
        ? 'transparent'
        : cell?.variant?.kind === 'special-object'
          ? 'var(--color-empty)'
        : 'var(--color-empty)',
    color: 'transparent',
    borderColor:
      cell?.variant?.kind === 'special-object'
        ? 'rgba(255,255,255,0.05)'
        : 'rgba(255,255,255,0.04)',
    boxShadow:
      cell?.variant?.kind === 'special-object'
        ? 'inset 0 1px 0 rgba(255,255,255,0.06)'
        : 'inset 0 1px 0 rgba(255,255,255,0.06)',
  };
  if (isSelected) {
    return {
      ...baseStyle,
      borderColor: 'rgba(251, 191, 36, 0.95)',
      boxShadow: '0 0 0 2px rgba(251, 191, 36, 0.9), inset 0 1px 0 rgba(255,255,255,0.12)',
    };
  }
  if (isValidTarget) {
    return {
      ...baseStyle,
      borderColor: 'rgba(56, 189, 248, 0.95)',
      boxShadow: '0 0 0 2px rgba(56, 189, 248, 0.78), inset 0 1px 0 rgba(255,255,255,0.08)',
    };
  }
  return baseStyle;
};

const syncToBoardRaw = (clearTimers = false) => {
  if (clearTimers) {
    clearAnimationTimers();
  }
  const nextSettledTiles = cells.value
    .filter((cell) => isRenderableTile(cell) && !effectHiddenIndices.value.has(cell.index))
    .map((cell) => createTileFromCell(cell));
  settledTiles.value = nextSettledTiles.map((tile) => ({ ...tile }));
  activeTiles.value = nextSettledTiles;
};

const fastForwardAnimations = () => {
  activeTiles.value = activeTiles.value
    .filter((tile) => !tile.isDying)
    .map((tile) => ({
      ...tile,
      isNew: false,
      isMerged: false,
      isPopActive: false,
      isHidden: false,
      isDying: false,
      isInterrupting: false,
    }));
  settledTiles.value = activeTiles.value.map((tile) => ({ ...tile }));
};

const revealMergedTiles = () => {
  activeTiles.value = activeTiles.value.map((tile) => {
    if (tile.isDying) {
      return { ...tile, isHidden: true };
    }
    if (tile.isMerged && tile.isHidden) {
      return { ...tile, isHidden: false, isPopActive: true };
    }
    return tile;
  });
};

const scheduleEffectsCleanup = (duration = 420, syncBoard = true) => {
  if (effectsCleanupTimeout) {
    window.clearTimeout(effectsCleanupTimeout);
  }
  effectsCleanupTimeout = window.setTimeout(() => {
    clearSpecialEffects();
    if (syncBoard) {
      syncToBoardRaw(false);
    }
    effectsCleanupTimeout = null;
  }, duration);
};

const scheduleMoveTimers = (followUp = null, hasConcurrentEffects = false) => {
  if (revealTimeout) {
    window.clearTimeout(revealTimeout);
  }
  revealTimeout = window.setTimeout(() => {
    revealMergedTiles();
    revealTimeout = null;
  }, 100);

  if (moveCleanupTimeout) {
    window.clearTimeout(moveCleanupTimeout);
  }
  moveCleanupTimeout = window.setTimeout(() => {
    fastForwardAnimations();
    if (!followUp) {
      if (!hasConcurrentEffects) {
        clearSpecialEffects();
      }
      syncToBoardRaw(false);
    }
    moveCleanupTimeout = null;
  }, followUp ? Math.max(300, Number(followUp.delayMs || 0)) : 300);

  if (followUp) {
    if (followUpTimeout) {
      window.clearTimeout(followUpTimeout);
    }
    const delayMs = Math.max(300, Number(followUp.delayMs || 0));
    followUpTimeout = window.setTimeout(() => {
      void runFollowUpAnimation(followUp);
      followUpTimeout = null;
    }, delayMs);
  }
};

const runFollowUpAnimation = async (followUp) => {
  const kind = String(followUp?.kind || '');
  if (kind === 'move') {
    const direction = String(followUp.direction || '').toLowerCase();
    const slideDistances = Array.isArray(followUp.slide_distances) ? followUp.slide_distances : [];
    const popPositions = Array.isArray(followUp.pop_positions) ? followUp.pop_positions : [];
    if (
      !['left', 'right', 'up', 'down'].includes(direction) ||
      slideDistances.length !== cellCount.value ||
      popPositions.length !== cellCount.value
    ) {
      clearSpecialEffects();
      syncToBoardRaw();
      return;
    }

    clearSpecialEffects();
    fastForwardAnimations();
    activeTiles.value = activeTiles.value.map((tile) => ({ ...tile, isInterrupting: true }));
    await nextTick();
    void boardRef.value?.offsetHeight;

    const vectors = {
      left: { x: -1, y: 0 },
      right: { x: 1, y: 0 },
      up: { x: 0, y: -1 },
      down: { x: 0, y: 1 },
    };
    const vector = vectors[direction];
    const nextTiles = [];

    activeTiles.value.forEach((tile) => {
      const oldIndex = tile.row * cols.value + tile.col;
      const distance = Number(slideDistances[oldIndex] || 0);
      let nextCol = tile.col;
      let nextRow = tile.row;

      if (distance > 0) {
        nextCol += vector.x * distance;
        nextRow += vector.y * distance;
        tile.col = nextCol;
        tile.row = nextRow;
        tile.index = nextRow * cols.value + nextCol;
      }

      const targetIndex = nextRow * cols.value + nextCol;
      if (Number(popPositions[targetIndex] || 0) === 1) {
        tile.isDying = true;
        if (!nextTiles.find((candidate) => candidate.index === targetIndex && candidate.isHidden)) {
          const mergedTile = createMergedAnimationTile(targetIndex, tile);
          if (mergedTile) {
            nextTiles.push(mergedTile);
          }
        }
      }

      tile.isInterrupting = false;
      nextTiles.push(tile);
    });

    activeTiles.value = nextTiles;
    scheduleMoveTimers(null, false);
    return;
  }

  if (kind !== 'effects') {
    clearSpecialEffects();
    syncToBoardRaw();
    return;
  }

  clearSpecialEffects();
  fastForwardAnimations();

  const transientEffects = [];
  const hiddenTargets = new Set();
  const hiddenSources = new Set();

  for (const effect of followUp.effects || []) {
    const type = String(effect?.type || '');
    if (!['column_swap_move', 'ring_rotate_move', 'gravity_fall_move'].includes(type)) {
      continue;
    }
    const fromIndex = Number(effect.fromIndex);
    const toIndex = Number(effect.toIndex);
    const value = Number(effect.value || 0);
    if (fromIndex < 0 || toIndex < 0 || value <= 0) {
      continue;
    }
    hiddenSources.add(fromIndex);
    hiddenTargets.add(toIndex);
    transientEffects.push({
      ...createSyntheticTile(fromIndex, value, {
        id: `effect-${tileIdCounter++}`,
        type,
        fromIndex,
        toIndex,
        targetRow: Math.floor(toIndex / cols.value),
        targetCol: toIndex % cols.value,
        durationMs: Number(effect.durationMs || 1250),
        animDurationMs: Number(effect.animDurationMs || effect.durationMs || 1250),
      }),
    });
  }

  activeTiles.value = activeTiles.value.map((tile) =>
    hiddenSources.has(tile.index) ? { ...tile, isHidden: true } : tile
  );
  effectHiddenIndices.value = hiddenTargets;
  await nextTick();
  specialEffects.value = transientEffects;
  scheduleEffectsCleanup(Number(followUp.durationMs || 1250));
};

const runSpecialEffects = async (effects) => {
  clearSpecialEffects();
  fastForwardAnimations();

  const transientEffects = [];
  const hiddenTargets = new Set();

  const addHiddenIndices = (effect) => {
    if (Array.isArray(effect?.hideIndices)) {
      effect.hideIndices.forEach((index) => hiddenTargets.add(Number(index)));
    }
    if (Number.isInteger(effect?.hideIndex)) {
      hiddenTargets.add(Number(effect.hideIndex));
    }
  };

  effects.forEach((effect) => {
    const type = String(effect?.type || '');
    if (!type) return;
    addHiddenIndices(effect);

    if (type === 'explosion' && Number.isInteger(effect.index)) {
      transientEffects.push({
        id: `effect-${tileIdCounter++}`,
        type,
        index: Number(effect.index),
        row: Math.floor(Number(effect.index) / cols.value),
        col: Number(effect.index) % cols.value,
        durationMs: Number(effect.durationMs || 500),
        animDurationMs: Number(effect.animDurationMs || effect.durationMs || 500),
        delayMs: Number(effect.delayMs || 0),
      });
      return;
    }

    if (
      ['airraid_fire_drop', 'airraid_explosion', 'giftbox_burst', 'factorization_burst', 'ice_stage_reveal'].includes(type) &&
      Number.isInteger(effect.index)
    ) {
      transientEffects.push({
        id: `effect-${tileIdCounter++}`,
        type,
        index: Number(effect.index),
        row: Math.floor(Number(effect.index) / cols.value),
        col: Number(effect.index) % cols.value,
        sprite: String(effect.sprite || ''),
        durationMs: Number(effect.durationMs || effect.animDurationMs || 420),
        animDurationMs: Number(effect.animDurationMs || effect.durationMs || 420),
        delayMs: Number(effect.delayMs || 0),
      });
      return;
    }

    if (type === 'grab' && Number.isInteger(effect.index)) {
      transientEffects.push({
        ...createSyntheticTile(Number(effect.index), Number(effect.value || 0), {
          id: `effect-${tileIdCounter++}`,
          type,
          durationMs: Number(effect.durationMs || 280),
          animDurationMs: Number(effect.animDurationMs || effect.durationMs || 280),
          delayMs: Number(effect.delayMs || 0),
        }),
      });
      return;
    }

    if (
      type === 'glove_move' &&
      Number.isInteger(effect.fromIndex) &&
      Number.isInteger(effect.toIndex)
    ) {
      const fromIndex = Number(effect.fromIndex);
      const toIndex = Number(effect.toIndex);
      hiddenTargets.add(toIndex);
      transientEffects.push({
        ...createSyntheticTile(fromIndex, Number(effect.value || 0), {
          id: `effect-${tileIdCounter++}`,
          type,
          fromIndex,
          toIndex,
          targetRow: Math.floor(toIndex / cols.value),
          targetCol: toIndex % cols.value,
          durationMs: Number(effect.durationMs || 340),
          animDurationMs: Number(effect.animDurationMs || effect.durationMs || 340),
          delayMs: Number(effect.delayMs || 0),
        }),
      });
      return;
    }

    if (
      type === 'object_slide' &&
      Number.isInteger(effect.fromIndex) &&
      Number.isInteger(effect.toIndex)
    ) {
      transientEffects.push(
        createObjectEffect({
          ...effect,
          durationMs: Number(effect.durationMs || 180),
          animDurationMs: Number(effect.animDurationMs || effect.durationMs || 180),
          delayMs: Number(effect.delayMs || 0),
        })
      );
      return;
    }

    if (type === 'twist' && Number.isInteger(effect.index)) {
      const index = Number(effect.index);
      const twistTiles = Array.isArray(effect.tiles) ? effect.tiles : [];
      twistTiles.forEach((tile) => {
        const fromIndex = Number(tile.fromIndex);
        const toIndex = Number(tile.toIndex);
        const value = Number(tile.value || 0);
        if (fromIndex < 0 || toIndex < 0 || value <= 0) return;
        hiddenTargets.add(toIndex);
        transientEffects.push({
          ...createSyntheticTile(fromIndex, value, {
            id: `effect-${tileIdCounter++}`,
            type: 'twist_move',
            fromIndex,
            toIndex,
            targetRow: Math.floor(toIndex / cols.value),
            targetCol: toIndex % cols.value,
            durationMs: Number(effect.durationMs || 260),
            animDurationMs: Number(effect.animDurationMs || effect.durationMs || 260),
            delayMs: Number(effect.delayMs || 0),
          }),
        });
      });
      transientEffects.push({
        id: `effect-${tileIdCounter++}`,
        type,
        index,
        row: Math.floor(index / cols.value),
        col: index % cols.value,
        durationMs: Number(effect.durationMs || 400),
        animDurationMs: Number(effect.animDurationMs || effect.durationMs || 400),
        delayMs: Number(effect.delayMs || 0),
      });
    }
  });

  effectHiddenIndices.value = hiddenTargets;
  syncToBoardRaw();
  await nextTick();
  specialEffects.value = transientEffects;
  const maxDuration = effects.reduce(
    (duration, effect) =>
      Math.max(
        duration,
        Number(effect?.delayMs || 0) + Number(effect?.durationMs || effect?.animDurationMs || 430)
      ),
    0
  );
  scheduleEffectsCleanup(maxDuration || 430, true);
};

const runConcurrentEffects = async (effects) => {
  clearSpecialEffects();

  const transientEffects = [];
  const hiddenTargets = new Set();

  const addHiddenIndices = (effect) => {
    if (Array.isArray(effect?.hideIndices)) {
      effect.hideIndices.forEach((index) => hiddenTargets.add(Number(index)));
    }
    if (Number.isInteger(effect?.hideIndex)) {
      hiddenTargets.add(Number(effect.hideIndex));
    }
  };

  effects.forEach((effect) => {
    const type = String(effect?.type || '');
    if (!type) return;
    addHiddenIndices(effect);

    if (
      ['explosion', 'airraid_fire_drop', 'airraid_explosion', 'giftbox_burst', 'factorization_burst', 'ice_stage_reveal'].includes(type) &&
      Number.isInteger(effect.index)
    ) {
      transientEffects.push({
        id: `effect-${tileIdCounter++}`,
        type,
        index: Number(effect.index),
        row: Math.floor(Number(effect.index) / cols.value),
        col: Number(effect.index) % cols.value,
        sprite: String(effect.sprite || ''),
        durationMs: Number(effect.durationMs || effect.animDurationMs || 420),
        animDurationMs: Number(effect.animDurationMs || effect.durationMs || 420),
        delayMs: Number(effect.delayMs || 0),
      });
      return;
    }

    if (
      type === 'object_slide' &&
      Number.isInteger(effect.fromIndex) &&
      Number.isInteger(effect.toIndex)
    ) {
      transientEffects.push(
        createObjectEffect({
          ...effect,
          durationMs: Number(effect.durationMs || 180),
          animDurationMs: Number(effect.animDurationMs || effect.durationMs || 180),
          delayMs: Number(effect.delayMs || 0),
        })
      );
    }
  });

  effectHiddenIndices.value = hiddenTargets;
  specialEffects.value = transientEffects;
  const maxDuration = effects.reduce(
    (duration, effect) =>
      Math.max(
        duration,
        Number(effect?.delayMs || 0) + Number(effect?.durationMs || effect?.animDurationMs || 430)
      ),
    0
  );
  scheduleEffectsCleanup(maxDuration || 430, (maxDuration || 430) > 320);
};

const runMoveAnimation = async () => {
  const metadata = props.metadata || {};
  const effects = Array.isArray(metadata.effects) ? metadata.effects : [];
  const direction = String(metadata.direction || '').toLowerCase();
  const slideDistances = Array.isArray(metadata.slide_distances) ? metadata.slide_distances : [];
  const popPositions = Array.isArray(metadata.pop_positions) ? metadata.pop_positions : [];
  const appearTile = metadata.appearTile || metadata.appear_tile || null;
  const followUp = metadata.followUp || null;

  const hasMoveAnimation =
    ['left', 'right', 'up', 'down'].includes(direction) &&
    slideDistances.length === cellCount.value &&
    popPositions.length === cellCount.value;

  if (!hasMoveAnimation && effects.length) {
    settledTiles.value = cells.value
      .filter((cell) => isRenderableTile(cell) && !effectHiddenIndices.value.has(cell.index))
      .map((cell) => createTileFromCell(cell));
    await runSpecialEffects(effects);
    return;
  }

  if (!hasMoveAnimation) {
    clearSpecialEffects();
    syncToBoardRaw(false);
    return;
  }

  clearSpecialEffects();
  if (settledTiles.value.length) {
    activeTiles.value = settledTiles.value.map((tile) => ({
      ...tile,
      isNew: false,
      isMerged: false,
      isHidden: false,
      isDying: false,
      isInterrupting: false,
    }));
  } else {
    fastForwardAnimations();
  }

  settledTiles.value = cells.value
    .filter((cell) => isRenderableTile(cell) && !effectHiddenIndices.value.has(cell.index))
    .map((cell) => createTileFromCell(cell));

  activeTiles.value.forEach((tile) => {
    tile.isInterrupting = true;
  });
  await nextTick();
  void boardRef.value?.offsetHeight;

  const vectors = {
    left: { x: -1, y: 0 },
    right: { x: 1, y: 0 },
    up: { x: 0, y: -1 },
    down: { x: 0, y: 1 },
  };
  const vector = vectors[direction];
  const nextTiles = [];

  activeTiles.value.forEach((tile) => {
    const oldIndex = tile.row * cols.value + tile.col;
    const distance = Number(slideDistances[oldIndex] || 0);
    let nextCol = tile.col;
    let nextRow = tile.row;

    if (distance > 0) {
      nextCol += vector.x * distance;
      nextRow += vector.y * distance;
      tile.col = nextCol;
      tile.row = nextRow;
      tile.index = nextRow * cols.value + nextCol;
    }

    const targetIndex = nextRow * cols.value + nextCol;
    if (Number(popPositions[targetIndex] || 0) === 1) {
      tile.isDying = true;
      if (!nextTiles.find((candidate) => candidate.index === targetIndex && candidate.isHidden)) {
        const mergedTile = createMergedAnimationTile(targetIndex, tile);
        if (mergedTile) {
          nextTiles.push(mergedTile);
        }
      }
    }

    tile.isInterrupting = false;
    nextTiles.push(tile);
  });

  if (appearTile && appearTile.index != null) {
    const appearIndex = Number(appearTile.index);
    const appearCell = cellMap.value.get(appearIndex);
    if (appearCell && isRenderableTile(appearCell)) {
      nextTiles.push(
        createTileFromCell(appearCell, {
          index: appearIndex,
          row: Math.floor(appearIndex / cols.value),
          col: appearIndex % cols.value,
          isNew: true,
        })
      );
    } else if (Number(appearTile.value || 0) > 0) {
      nextTiles.push(
        createSyntheticTile(appearIndex, Number(appearTile.value), {
          id: `minigame-appear-${tileIdCounter++}`,
          index: appearIndex,
          row: Math.floor(appearIndex / cols.value),
          col: appearIndex % cols.value,
          isNew: true,
        })
      );
    }
  }

  activeTiles.value = nextTiles;
  if (effects.length) {
    await runConcurrentEffects(effects);
  }
  scheduleMoveTimers(followUp, effects.length > 0);
};

watch(
  [() => props.board, () => props.metadata, () => props.shape, () => props.view],
  async () => {
    await runMoveAnimation();
  },
  { deep: true, immediate: true }
);

onBeforeUnmount(() => {
  clearAnimationTimers();
});

const getTilePosStyle = (tile) => ({
  width: `calc((100% - ${(cols.value - 1)} * var(--grid-gap)) / ${cols.value})`,
  height: `calc((100% - ${(rows.value - 1)} * var(--grid-gap)) / ${rows.value})`,
  transform: `translate(calc(${tile.col} * (100% + var(--grid-gap))), calc(${tile.row} * (100% + var(--grid-gap))))`,
});

const getSubgridCenterStyle = (row, col) => ({
  left: `calc(${col + 1} * ((100% - ${(cols.value - 1)} * var(--grid-gap)) / ${cols.value}) + ${col} * var(--grid-gap) + (var(--grid-gap) / 2))`,
  top: `calc(${row + 1} * ((100% - ${(rows.value - 1)} * var(--grid-gap)) / ${rows.value}) + ${row} * var(--grid-gap) + (var(--grid-gap) / 2))`,
});

const getCellSlotStyle = (cell) => getTilePosStyle(cell);

const getTileInnerStyle = (tile) => {
  const isValidTarget =
    props.interaction?.mode !== 'twist' && interactionValidTargets.value.has(tile.index);
  const isSelected = interactionSelected.value.has(tile.index);
  return {
    background: tile.background,
    color: tile.color,
    borderColor: isSelected
      ? 'rgba(251, 191, 36, 0.95)'
      : isValidTarget
        ? 'rgba(56, 189, 248, 0.95)'
        : tile.borderColor,
    boxShadow: isSelected
      ? '0 0 0 2px rgba(251, 191, 36, 0.9), 0 0 18px rgba(251, 191, 36, 0.34), inset 0 1px 0 rgba(255,255,255,0.1)'
      : isValidTarget
        ? '0 0 0 2px rgba(56, 189, 248, 0.76), 0 0 16px rgba(56, 189, 248, 0.24), inset 0 1px 0 rgba(255,255,255,0.08)'
        : tile.boxShadow || 'inset 0 1px 0 rgba(255, 255, 255, 0.08)',
  };
};

const getTileLabelStyle = (tile) => {
  const textLength = String(tile.displayText || '').length || 1;
  let fontSize = 'clamp(1rem, 8cqw, 2.5rem)';
  if (textLength >= 5) {
    fontSize = 'clamp(0.8rem, 5.2cqw, 1.55rem)';
  } else if (textLength === 4) {
    fontSize = 'clamp(0.92rem, 5.8cqw, 1.85rem)';
  } else if (textLength === 3) {
    fontSize = 'clamp(1rem, 6.8cqw, 2.15rem)';
  }
  return {
    fontSize: `calc(${fontSize} * var(--tile-font-scale, 1) * var(--shape-text-scale, 1))`,
    color:
      tile?.variant?.kind === 'mystery-hidden' || tile?.variant?.kind === 'tilebg'
        ? '#fff'
        : undefined,
    textShadow:
      tile?.variant?.kind === 'mystery-hidden' || tile?.variant?.kind === 'tilebg'
        ? '0 2px 6px rgba(15, 23, 42, 0.35)'
        : undefined,
  };
};

const getEffectWrapperStyle = (effect) => {
  const baseVars = {
    '--effect-duration-ms': Number(effect?.animDurationMs || effect?.durationMs || 420),
    '--effect-delay-ms': Number(effect?.delayMs || 0),
  };
  if (effect.type === 'twist') {
    const center = getSubgridCenterStyle(Number(effect.row || 0), Number(effect.col || 0));
    return {
        ...baseVars,
        ...center,
        width: `calc((((100% - ${(cols.value - 1)} * var(--grid-gap)) / ${cols.value}) * 2) + var(--grid-gap))`,
        height: `calc((((100% - ${(rows.value - 1)} * var(--grid-gap)) / ${rows.value}) * 2) + var(--grid-gap))`,
        transform: 'translate(-50%, -50%)',
      };
    }

  const baseStyle = getTilePosStyle(effect);
  if (
    ['glove_move', 'twist_move', 'column_swap_move', 'ring_rotate_move', 'gravity_fall_move', 'object_slide'].includes(
      effect.type
    )
  ) {
    return {
      ...baseStyle,
      ...baseVars,
      '--delta-x': Number(effect.targetCol - effect.col),
      '--delta-y': Number(effect.targetRow - effect.row),
      '--object-slide-end-opacity': effect.type === 'object_slide' && effect.fadeOutAtEnd ? '0' : '1',
    };
  }
  if (effect.type === 'airraid_fire_drop') {
    return {
      ...baseStyle,
      ...baseVars,
      '--drop-start-y': '-360px',
    };
  }
  return {
    ...baseStyle,
    ...baseVars,
  };
};

const getTwistHotspotStyle = (hotspot) => getSubgridCenterStyle(hotspot.row, hotspot.col);

const getEffectInnerStyle = (effect) => {
  if (
    ['explosion', 'airraid_fire_drop', 'airraid_explosion', 'twist', 'giftbox_burst', 'factorization_burst', 'ice_stage_reveal'].includes(effect.type)
  ) {
    return {};
  }
  if (effect.type === 'object_slide') {
    return {
      background: 'transparent',
      color: 'transparent',
      borderColor: 'transparent',
      boxShadow: 'none',
    };
  }
  return getTileInnerStyle(effect);
};
</script>

<style scoped>
.board {
  --shape-density-scale: 1;
  --shape-text-scale: 1;
  --padding: calc(2.5cqw * var(--shape-density-scale));
  --grid-gap: calc(2.5cqw * var(--shape-density-scale));
}

.has-custom-cursor,
.has-custom-cursor * {
  cursor: none !important;
}

.minigame-board-grid,
.minigame-active-layer,
.minigame-small-label-layer,
.minigame-effects-layer {
  position: absolute;
  top: var(--padding);
  left: var(--padding);
  right: var(--padding);
  bottom: var(--padding);
}

.minigame-board-grid {
  display: grid;
  gap: var(--grid-gap);
  z-index: 0;
}

.minigame-active-layer {
  pointer-events: none;
  z-index: 10;
}

.minigame-small-label-layer {
  pointer-events: none;
  z-index: 15;
}

.minigame-effects-layer {
  pointer-events: none;
  z-index: 20;
}

.minigame-small-label-slot {
  position: absolute;
  top: 0;
  left: 0;
}

.minigame-hotspots-layer {
  position: absolute;
  inset: var(--padding);
  z-index: 25;
}

.minigame-cell,
.minigame-tile-inner {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: calc(0.36rem + 0.14rem * var(--shape-density-scale));
  border: 1px solid transparent;
  font-weight: 900;
  line-height: 1;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
  overflow: hidden;
}

.minigame-cell {
  aspect-ratio: 1 / 1;
  padding: 0;
}

.minigame-cell.is-hole-cell {
  opacity: 0;
  box-shadow: none !important;
  border-color: transparent !important;
}

.minigame-tile {
  pointer-events: none;
  position: absolute;
  top: 0;
  left: 0;
  transition: transform 0.1s ease-in-out;
}

.minigame-effect {
  pointer-events: none;
  position: absolute;
  top: 0;
  left: 0;
  --effect-duration-ms: 420;
  --effect-delay-ms: 0;
}

.twist-hotspot {
  position: absolute;
  width: clamp(30px, 10cqw, 44px);
  height: clamp(30px, 10cqw, 44px);
  transform: translate(-50%, -50%);
  border-radius: 999px;
  border: none;
  padding: 0;
  background: transparent;
  box-shadow: none;
  touch-action: manipulation;
}

.twist-hotspot::before {
  content: '';
  position: absolute;
  inset: 50%;
  width: clamp(16px, 5.2cqw, 24px);
  height: clamp(16px, 5.2cqw, 24px);
  transform: translate(-50%, -50%);
  border-radius: 999px;
  border: 1px solid rgba(56, 189, 248, 0.76);
  background: radial-gradient(circle, rgba(56, 189, 248, 0.44), rgba(56, 189, 248, 0.14) 64%, rgba(56, 189, 248, 0) 100%);
  box-shadow: 0 0 16px rgba(56, 189, 248, 0.22);
}

.twist-hotspot:hover::before,
.twist-hotspot:focus-visible::before {
  border-color: rgba(56, 189, 248, 0.94);
  box-shadow: 0 0 20px rgba(56, 189, 248, 0.32);
}

.cursor-preview {
  pointer-events: none;
  position: absolute;
  z-index: 35;
  width: clamp(28px, 9cqw, 52px);
  height: clamp(28px, 9cqw, 52px);
  transform: translate(-28%, -22%);
}

.cursor-preview-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  user-select: none;
  -webkit-user-drag: none;
  filter: drop-shadow(0 3px 10px rgba(15, 23, 42, 0.25));
}

.minigame-tile-inner {
  width: 100%;
  height: 100%;
  transition: background 0.16s ease, color 0.16s ease, border-color 0.16s ease, opacity 0s;
}

.minigame-cell-label {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  padding: 0.25rem;
  font-weight: 900;
  line-height: 1;
  text-align: center;
  z-index: 2;
}

.label-mystery-hidden {
  letter-spacing: 0.04em;
  text-shadow: 0 2px 6px rgba(15, 23, 42, 0.35);
}

.label-frozen {
  text-shadow: 0 1px 6px rgba(226, 232, 240, 0.18);
}

.minigame-cell-small {
  position: absolute;
  right: 0.4rem;
  bottom: 0.35rem;
  font-size: calc(clamp(0.45rem, 0.75vw, 0.68rem) * var(--shape-text-scale));
  font-weight: 900;
  opacity: 0.9;
  color: var(--text-secondary);
  text-shadow: 0 1px 3px rgba(15, 23, 42, 0.18);
  z-index: 2;
}

.minigame-cell-overlay {
  position: absolute;
  left: 0.4rem;
  top: 0.35rem;
  font-size: calc(clamp(0.45rem, 0.75vw, 0.72rem) * var(--shape-text-scale));
  font-weight: 900;
  opacity: 0.9;
  color: var(--text-secondary);
  text-shadow: 0 1px 3px rgba(15, 23, 42, 0.18);
  z-index: 2;
}

.minigame-cover-stack {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.minigame-cover-sprite {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  user-select: none;
  -webkit-user-drag: none;
}

.cover-sprite-tilebg {
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  filter: drop-shadow(0 2px 6px rgba(15, 23, 42, 0.18));
}

.cover-sprite-portal {
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.22));
}

.cover-sprite-target {
  animation: target-breathe 1.6s ease-in-out infinite;
  filter: drop-shadow(0 0 16px rgba(248, 113, 113, 0.18));
}

.cover-sprite-bomb {
  inset: 8%;
  width: 84%;
  height: 84%;
  filter: drop-shadow(0 2px 8px rgba(15, 23, 42, 0.2));
}

.cover-sprite-giftbox {
  inset: 7%;
  width: 86%;
  height: 86%;
  filter: drop-shadow(0 2px 8px rgba(15, 23, 42, 0.18));
}

.cover-sprite-crystal1 {
  inset: 10% auto auto 54%;
  width: 26%;
  height: 26%;
  animation: crystal-flicker 2.2s ease-in-out infinite;
}

.cover-sprite-crystal3 {
  inset: 12% auto auto 42%;
  width: 16.666%;
  height: 16.666%;
  animation: crystal-flicker 2.2s ease-in-out infinite;
}

.cover-sprite-crystal2 {
  inset: auto auto 16% 6%;
  width: 25%;
  height: 25%;
  animation: crystal-flicker 2.2s ease-in-out infinite;
}

.cover-sprite-ice-overlay {
  inset: 0;
  width: 100%;
  height: 100%;
  animation: ice-shimmer 2.6s ease-in-out infinite;
}

.cover-sprite-icetrap0 {
  inset: -8% 0 auto 0;
  width: 100%;
  height: 25%;
  animation: ice-shimmer 2.6s ease-in-out infinite;
}

.cover-sprite-icetrap {
  inset: auto -10% 14% -10%;
  width: 120%;
  height: 50%;
  animation: ice-shimmer 2.6s ease-in-out infinite;
}

.cover-sprite-fire {
  animation: fire-drop 0.38s ease-in forwards;
}

.no-transition {
  transition: none !important;
}

.anim-new {
  animation: appear 0.2s ease backwards;
}

.anim-merged {
  animation: pop 0.2s ease backwards;
}

.effect-explosion {
  width: 100%;
  height: 100%;
  animation: explode-burst calc(var(--effect-duration-ms) * 1ms) ease-out forwards;
  animation-delay: calc(var(--effect-delay-ms) * 1ms);
}

.effect-grab {
  animation: grab-lift calc(var(--effect-duration-ms) * 1ms) ease-out forwards;
  animation-delay: calc(var(--effect-delay-ms) * 1ms);
}

.effect-glove-move {
  animation: glove-fly calc(var(--effect-duration-ms) * 1ms) ease-in-out forwards;
  animation-delay: calc(var(--effect-delay-ms) * 1ms);
}

.effect-column-swap-move {
  animation: column-swap-fly calc(var(--effect-duration-ms) * 1ms) cubic-bezier(0.22, 0.9, 0.3, 1) forwards;
  animation-delay: calc(var(--effect-delay-ms) * 1ms);
}

.effect-ring-rotate-move {
  animation: ring-rotate-fly calc(var(--effect-duration-ms) * 1ms) cubic-bezier(0.16, 0.88, 0.24, 1) forwards;
  animation-delay: calc(var(--effect-delay-ms) * 1ms);
}

.effect-gravity-fall-move {
  animation: gravity-fall-fly calc(var(--effect-duration-ms) * 1ms) cubic-bezier(0.18, 0.9, 0.24, 1) forwards;
  animation-delay: calc(var(--effect-delay-ms) * 1ms);
}

.effect-twist-move {
  animation: twist-tile-move calc(var(--effect-duration-ms) * 1ms) cubic-bezier(0.22, 0.9, 0.3, 1) forwards;
  animation-delay: calc(var(--effect-delay-ms) * 1ms);
}

.effect-twist {
  width: 100%;
  height: 100%;
  animation: twist-spin calc(var(--effect-duration-ms) * 1ms) cubic-bezier(0.22, 0.9, 0.3, 1) forwards;
  animation-delay: calc(var(--effect-delay-ms) * 1ms);
  transform-origin: center center;
}

.effect-object-slide {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  border-radius: 0.5rem;
  animation: object-slide-fly calc(var(--effect-duration-ms) * 1ms) cubic-bezier(0.18, 0.88, 0.24, 1) forwards;
  animation-delay: calc(var(--effect-delay-ms) * 1ms);
}

.effect-airraid-fire-drop {
  width: 100%;
  height: 100%;
  animation: fire-drop-attack calc(var(--effect-duration-ms) * 1ms) cubic-bezier(0.3, 0.1, 0.85, 0.18) forwards;
  animation-delay: calc(var(--effect-delay-ms) * 1ms);
}

.effect-airraid-explosion,
.effect-giftbox-burst,
.effect-factorization-burst,
.effect-ice-stage-reveal {
  width: 100%;
  height: 100%;
  animation-delay: calc(var(--effect-delay-ms) * 1ms);
}

.effect-airraid-explosion {
  animation: airraid-impact calc(var(--effect-duration-ms) * 1ms) ease-out forwards;
}

.effect-giftbox-burst,
.effect-factorization-burst {
  animation: burst-reveal calc(var(--effect-duration-ms) * 1ms) cubic-bezier(0.22, 0.9, 0.3, 1) forwards;
}

.effect-ice-stage-reveal {
  animation: ice-reveal calc(var(--effect-duration-ms) * 1ms) ease-out forwards;
}

.effect-image:not(.minigame-cover-sprite) {
  width: 100%;
  height: 100%;
  object-fit: contain;
  user-select: none;
  -webkit-user-drag: none;
}

.effect-explosion-image {
  filter: drop-shadow(0 0 18px rgba(251, 191, 36, 0.28));
}

.effect-glove-image {
  position: absolute;
  inset: -12%;
  width: 124%;
  height: 124%;
  object-fit: contain;
  opacity: 0.95;
  filter: drop-shadow(0 3px 10px rgba(15, 23, 42, 0.28));
}

.effect-twist-image {
  filter: drop-shadow(0 0 18px rgba(56, 189, 248, 0.26));
}

.effect-fire-image {
  filter: drop-shadow(0 4px 12px rgba(248, 113, 113, 0.24));
}

.effect-giftbox-image,
.effect-factorization-image {
  filter: drop-shadow(0 0 14px rgba(15, 23, 42, 0.18));
}

.effect-object-image {
  filter: drop-shadow(0 3px 10px rgba(15, 23, 42, 0.24));
}

.effect-ice-reveal-image {
  filter: drop-shadow(0 0 10px rgba(191, 219, 254, 0.24));
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

@keyframes explode-burst {
  0% {
    transform: scale(0.28);
    opacity: 0.95;
  }
  70% {
    transform: scale(1.18);
    opacity: 0.78;
  }
  100% {
    transform: scale(1.45);
    opacity: 0;
  }
}

@keyframes grab-lift {
  0% {
    transform: scale(1) translateY(0);
    opacity: 0.96;
  }
  45% {
    transform: scale(1.08) translateY(-10%);
    opacity: 1;
  }
  100% {
    transform: scale(1) translateY(-16%);
    opacity: 0;
  }
}

@keyframes glove-fly {
  0% {
    transform: translate(0, 0) scale(1.04);
    opacity: 0.98;
  }
  100% {
    transform: translate(
      calc(var(--delta-x) * (100% + var(--grid-gap))),
      calc(var(--delta-y) * (100% + var(--grid-gap)))
    ) scale(1);
    opacity: 0;
  }
}

@keyframes column-swap-fly {
  0% {
    transform: translate(0, 0) scale(1);
    opacity: 1;
  }
  100% {
    transform: translate(
      calc(var(--delta-x) * (100% + var(--grid-gap))),
      calc(var(--delta-y) * (100% + var(--grid-gap)))
    ) scale(1);
    opacity: 1;
  }
}

@keyframes ring-rotate-fly {
  0% {
    transform: translate(0, 0) scale(1);
    opacity: 1;
  }
  100% {
    transform: translate(
      calc(var(--delta-x) * (100% + var(--grid-gap))),
      calc(var(--delta-y) * (100% + var(--grid-gap)))
    ) scale(1);
    opacity: 1;
  }
}

@keyframes gravity-fall-fly {
  0% {
    transform: translate(0, 0) scale(1);
    opacity: 1;
  }
  100% {
    transform: translate(
      calc(var(--delta-x) * (100% + var(--grid-gap))),
      calc(var(--delta-y) * (100% + var(--grid-gap)))
    ) scale(1);
    opacity: 1;
  }
}

@keyframes twist-spin {
  0% {
    transform: rotate(0deg) scale(0.92);
    opacity: 0.18;
  }
  20% {
    opacity: 0.92;
  }
  100% {
    transform: rotate(90deg) scale(1.02);
    opacity: 0;
  }
}

@keyframes twist-tile-move {
  0% {
    transform: translate(0, 0) scale(1);
    opacity: 0.98;
  }
  100% {
    transform: translate(
      calc(var(--delta-x) * (100% + var(--grid-gap))),
      calc(var(--delta-y) * (100% + var(--grid-gap)))
    ) scale(1);
    opacity: 1;
  }
}

@keyframes object-slide-fly {
  0% {
    transform: translate(0, 0);
    opacity: 1;
  }
  85% {
    transform: translate(
      calc(var(--delta-x) * (100% + var(--grid-gap))),
      calc(var(--delta-y) * (100% + var(--grid-gap)))
    );
    opacity: 1;
  }
  100% {
    transform: translate(
      calc(var(--delta-x) * (100% + var(--grid-gap))),
      calc(var(--delta-y) * (100% + var(--grid-gap)))
    );
    opacity: var(--object-slide-end-opacity, 1);
  }
}

@keyframes fire-drop-attack {
  0% {
    transform: translateY(var(--drop-start-y)) scale(0.92);
    opacity: 0.88;
  }
  100% {
    transform: translateY(0) scale(1);
    opacity: 1;
  }
}

@keyframes airraid-impact {
  0% {
    transform: scale(0.34);
    opacity: 0.95;
  }
  70% {
    transform: scale(1.08);
    opacity: 0.86;
  }
  100% {
    transform: scale(1.2);
    opacity: 0;
  }
}

@keyframes burst-reveal {
  0% {
    transform: scale(0.4);
    opacity: 0.96;
  }
  60% {
    transform: scale(1.08);
    opacity: 0.88;
  }
  100% {
    transform: scale(1.18);
    opacity: 0;
  }
}

@keyframes ice-reveal {
  0% {
    transform: scale(0.8);
    opacity: 0;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
}

@keyframes target-breathe {
  0%, 100% {
    transform: scale(0.84);
    opacity: 0.78;
  }
  50% {
    transform: scale(1);
    opacity: 1;
  }
}

@keyframes object-hover {
  0%, 100% {
    transform: translateY(0);
  }
  50% {
    transform: translateY(-4%);
  }
}

@keyframes giftbox-hover {
  0%, 100% {
    transform: translateY(0) rotate(0deg);
  }
  25% {
    transform: translateY(-3%) rotate(-3deg);
  }
  75% {
    transform: translateY(-2%) rotate(3deg);
  }
}

@keyframes crystal-flicker {
  0%, 100% {
    opacity: 0.92;
  }
  50% {
    opacity: 1;
  }
}

@keyframes ice-shimmer {
  0%, 100% {
    opacity: 0.84;
  }
  50% {
    opacity: 1;
  }
}

@keyframes fire-drop {
  0% {
    transform: translateY(-18%) scale(0.92);
    opacity: 0.88;
  }
  100% {
    transform: translateY(0) scale(1);
    opacity: 1;
  }
}
</style>
