<template>
  <div class="guide-document">
    <header class="guide-document-header">
      <h1>{{ document.title }}</h1>
    </header>
    <template
      v-for="(block, index) in document.blocks"
      :key="block.id || `${block.type}-${block.paragraph_index ?? index}`"
    >
      <component
        :is="headingTag(block.level)"
        v-if="block.type === 'heading'"
        :id="block.id"
        class="guide-heading"
        :class="`guide-heading-level-${block.level}`"
      >
        {{ block.text }}
      </component>

      <p v-else-if="block.type === 'paragraph'" class="guide-paragraph">
        {{ block.text }}
      </p>

      <figure v-else-if="block.type === 'figure'" class="guide-figure">
        <div
          class="guide-figure-canvas"
          :style="figureStyle(block)"
          @click="openSingleBoard(block)"
        >
          <img
            :src="block.src"
            :alt="block.caption || block.image_id"
            :width="block.width"
            :height="block.height"
            loading="lazy"
            decoding="async"
            class="guide-figure-image"
          />
          <button
            v-for="board in block.boards || []"
            :key="board.board_id"
            type="button"
            class="guide-board-hotspot"
            :style="hotspotStyle(board, block)"
            :title="`${openBoardLabel}: ${board.hex}`"
            :aria-label="`${openBoardLabel} ${board.board_id}: ${board.hex}`"
            @click.stop="openBoard(board)"
          >
            <span class="guide-board-hotspot-label" aria-hidden="true">
              {{ board.board_id }}
            </span>
          </button>
        </div>
        <figcaption v-if="block.caption" class="guide-figure-caption">
          {{ block.caption }}
        </figcaption>
      </figure>
    </template>
  </div>
</template>

<script setup>
const props = defineProps({
  document: { type: Object, required: true },
  openBoardLabel: { type: String, default: 'Open in Trainer' },
});

const emit = defineEmits(['open-board']);
const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));
const headingTag = (level) => `h${clamp(Number(level) || 2, 1, 6)}`;

const figureStyle = (figure) => ({
  width: `${Math.max(1, Number(figure.width) || 1)}px`,
  maxWidth: '100%',
  aspectRatio: `${Math.max(1, Number(figure.width) || 1)} / ${Math.max(1, Number(figure.height) || 1)}`,
});

const hotspotStyle = (board, figure) => {
  const [x = 0, y = 0, width = 0, height = 0] = board?.bbox || [];
  const imageWidth = Math.max(1, Number(figure?.width) || 1);
  const imageHeight = Math.max(1, Number(figure?.height) || 1);
  return {
    left: `${clamp(Number(x) / imageWidth * 100, 0, 100)}%`,
    top: `${clamp(Number(y) / imageHeight * 100, 0, 100)}%`,
    width: `${clamp(Number(width) / imageWidth * 100, 0, 100)}%`,
    height: `${clamp(Number(height) / imageHeight * 100, 0, 100)}%`,
  };
};

const openBoard = (board) => {
  emit('open-board', board, props.document.id, props.document.trainer);
};

const openSingleBoard = (figure) => {
  if (figure?.boards?.length === 1) {
    openBoard(figure.boards[0]);
  }
};
</script>

<style scoped>
.guide-document {
  max-width: 56rem;
  color: var(--text-main);
  font-family: var(--font-stack-system);
  line-height: 1.75;
}

.guide-document-header {
  margin-bottom: 2.5rem;
  border-bottom: 1px solid var(--border-main);
  padding-bottom: 1rem;
}

.guide-document-header h1 {
  margin: 0;
  color: var(--text-main);
  font-size: calc(2.25rem * var(--ui-scale));
  font-weight: 900;
  letter-spacing: 0;
}

.guide-heading {
  color: var(--text-main);
  letter-spacing: 0;
  scroll-margin-top: 1.5rem;
}

.guide-heading-level-1 {
  margin: 3rem 0 1.5rem;
  border-bottom: 2px solid var(--accent);
  padding-bottom: 0.65rem;
  font-size: calc(2rem * var(--ui-scale));
  font-weight: 900;
}

.guide-heading-level-2 {
  margin: 2.25rem 0 1rem;
  border-bottom: 1px solid var(--border-main);
  padding-bottom: 0.45rem;
  font-size: calc(1.45rem * var(--ui-scale));
  font-weight: 850;
}

.guide-heading-level-3,
.guide-heading-level-4 {
  margin: 1.75rem 0 0.75rem;
  color: var(--accent);
  font-size: calc(1.15rem * var(--ui-scale));
  font-weight: 800;
}

.guide-paragraph {
  margin: 0.75rem 0 1.25rem;
  white-space: pre-wrap;
}

.guide-figure {
  margin: 1.75rem 0 2rem;
}

.guide-figure-canvas {
  position: relative;
  overflow: hidden;
  background: white;
}

.guide-figure-image {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: fill;
}

.guide-board-hotspot {
  position: absolute;
  border: 2px solid transparent;
  background: transparent;
  cursor: pointer;
  outline: none;
  transition: border-color 120ms ease, background-color 120ms ease;
}

.guide-board-hotspot:hover,
.guide-board-hotspot:focus-visible {
  border-color: var(--accent);
  background: color-mix(in srgb, var(--accent) 10%, transparent);
}

.guide-board-hotspot-label {
  position: absolute;
  top: 0;
  left: 0;
  display: none;
  max-width: 100%;
  overflow: hidden;
  background: var(--accent);
  padding: 0.1rem 0.25rem;
  color: white;
  font-size: 0.625rem;
  line-height: 1.15;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.guide-board-hotspot:hover .guide-board-hotspot-label,
.guide-board-hotspot:focus-visible .guide-board-hotspot-label {
  display: block;
}

.guide-figure-caption {
  margin-top: 0.5rem;
  color: var(--text-secondary);
  font-size: calc(0.8rem * var(--ui-scale));
  font-weight: 700;
}
</style>
