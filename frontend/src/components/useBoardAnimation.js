import { ref, watch, nextTick, onUnmounted } from 'vue';
import { boardFrameRenderMode, cloneBoard } from './boardFrame.js';

// Shared by the main board and human boards: consume explicit frames, snap interrupted
// transitions to their committed target, then slide, reveal merges/spawns and settle.
export function animationLayoutTarget(surface, fallback = document.body) {
  return surface?.querySelector?.('.moving-tile, .tile') || surface || fallback;
}

export function useBoardAnimation(props, boardViewport, viewportSignature, animationSurface = null) {
  let tileIdCounter = 0;
  const activeTiles = ref([]);
  let animTimeout = null;
  let revealMergeTimeout = null;
  let revealAppearTimeout = null;
  let animationEpoch = 0;
  let lastConsumedFrameRevision = null;
  let settledBoard = cloneBoard(props.frame?.toBoard);
  const MERGE_GLOW_MIN_VALUE = 2048;
  const MERGE_GLOW_STEPS = 5;
  function isVariantWallValue(value) {
    return props.isVariant && Number(value) === 32768;
  }

  function isVariantNonMergingValue(value) {
    return props.isVariant && Number(value) === 16384;
  }

  function shouldRenderAsActiveTile(value) {
    return Number(value) > 0 && !isVariantWallValue(value);
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
      // Commit the transition-free tile positions before sliding. Restrict the
      // synchronous layout read to this board instead of invalidating the page.
      void animationLayoutTarget(animationSurface?.value).offsetHeight;

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


  return { activeTiles };
}
