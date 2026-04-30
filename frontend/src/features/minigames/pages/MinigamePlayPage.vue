<template>
  <div class="page-root pt-6">
    <div ref="playRootRef" class="relative w-full max-w-lg flex flex-col items-center">
      <div v-if="pageEffects.length" class="minigame-page-effects-layer">
        <img
          v-for="effect in pageEffects"
          :key="effect.id"
          :src="effect.src"
          alt=""
          draggable="false"
          class="minigame-page-effect"
          :style="effect.style"
        />
      </div>
      <div class="flex justify-between w-full mb-6 items-center gap-4">
        <div class="flex items-center gap-4 min-w-0">
          <div class="min-w-0">
            <h1 class="minigame-title text-text-main font-bold leading-[0.94] font-[Cambria,serif]">
              {{ state.title }}
            </h1>
          </div>
          <span :class="connectionBadgeClass">{{ $t(`status.${wsStatus}`) }}</span>
        </div>
        <div class="flex space-x-2">
          <div class="bg-board-bg w-[122px] h-[56px] flex flex-col items-center justify-center rounded-md relative shadow-sm transition-all duration-300">
            <span class="text-text-secondary ui-caption font-black uppercase leading-none mb-1 tracking-tight">{{ $t('labels.score') }}</span>
            <span class="font-black text-white leading-none tabular-nums" style="font-size: calc(22px * var(--ui-scale));">{{ state.score }}</span>
          </div>
          <div class="bg-board-bg w-[122px] h-[56px] flex flex-col items-center justify-center rounded-md shadow-sm transition-all duration-300">
            <span class="text-text-secondary ui-caption font-black uppercase leading-none mb-1 tracking-tight">{{ $t('labels.best') }}</span>
            <span class="font-black text-white leading-none tabular-nums" style="font-size: calc(22px * var(--ui-scale));">{{ state.best }}</span>
          </div>
        </div>
      </div>

      <div class="w-full mb-4">
        <div class="flex w-full space-x-2">
          <button type="button" class="flex-1 bg-btn-bg text-white font-bold py-2 px-2 rounded hover:bg-btn-hover ui-body whitespace-nowrap" @click="$emit('new-game')">
            {{ $t('buttons.newGame') }}
          </button>
          <button type="button" class="flex-1 bg-btn-bg text-white font-bold py-2 px-2 rounded hover:bg-btn-hover ui-body whitespace-nowrap" @click="$emit('info')">
            {{ $t('minigames.play.info') }}
          </button>
          <button type="button" class="flex-1 bg-btn-bg text-white font-bold py-2 px-2 rounded hover:bg-btn-hover ui-body whitespace-nowrap" @click="$emit('back-menu')">
            {{ $t('minigames.play.backToMenu') }}
          </button>
        </div>
      </div>

      <div v-if="hasHudPanels" class="w-full mb-4">
        <MinigameHud :hud="state.hud" @custom-action="$emit('custom-action', $event)" />
      </div>

      <div class="w-full">
        <MinigameBoard
          :board="state.board"
          :shape="state.shape"
          :view="state.view"
          :metadata="state.animation"
          :interaction="state.interaction"
          @cell-click="$emit('cell-click', $event)"
        />
      </div>

      <div class="w-full mt-6">
        <PowerUpBar
          :powerups="state.powerups"
          :interaction="state.interaction"
          @use="$emit('use-powerup', $event)"
          @cancel="$emit('cancel-interaction')"
        />
      </div>

      <div v-if="toastMessage" class="w-full mt-3">
        <div class="badge-state badge-state-running w-full justify-center">
          {{ toastMessage }}
        </div>
      </div>

      <MinigameOverlay
        :overlay="overlay"
        @close="$emit('close-overlay')"
        @new-game="$emit('new-game')"
        @back-menu="$emit('back-menu')"
      />
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref, watch } from 'vue';

import MinigameBoard from '../components/MinigameBoard.vue';
import MinigameHud from '../components/MinigameHud.vue';
import MinigameOverlay from '../components/MinigameOverlay.vue';
import PowerUpBar from '../components/PowerUpBar.vue';
import { getMinigameAssetUrl } from '../../../services/runtime/backendUrl';

const props = defineProps({
  state: {
    type: Object,
    required: true,
  },
  overlay: {
    type: Object,
    required: true,
  },
  toastMessage: {
    type: String,
    default: '',
  },
  wsStatus: {
    type: String,
    default: 'connecting',
  },
  connectionBadgeClass: {
    type: String,
    default: 'badge-base badge-connection-pending',
  },
});

defineEmits([
  'back-menu',
  'new-game',
  'info',
  'use-powerup',
  'cancel-interaction',
  'cell-click',
  'custom-action',
  'close-overlay',
]);

const hasHudPanels = computed(() =>
  (props.state?.hud?.customPanels || []).some(
    (panel) => panel?.type !== 'patternText' && panel?.type !== 'targetPattern'
  )
);

const playRootRef = ref(null);
const pageEffects = ref([]);
let sparkleTimer = null;

const clearSparkleTimer = () => {
  if (sparkleTimer) {
    window.clearInterval(sparkleTimer);
    sparkleTimer = null;
  }
};

const pushPageEffect = ({ src, x, y, width, height, durationMs = 1000 }) => {
  const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  pageEffects.value = pageEffects.value.concat({
    id,
    src,
    style: {
      left: `${x}px`,
      top: `${y}px`,
      width: `${width}px`,
      height: `${height}px`,
      '--page-effect-duration-ms': Number(durationMs),
    },
  });
  window.setTimeout(() => {
    pageEffects.value = pageEffects.value.filter((effect) => effect.id !== id);
  }, Number(durationMs) + 30);
};

const maybeSpawnIceSparkle = () => {
  if (props.state?.gameId !== 'ice-age') return;
  if (!playRootRef.value || Math.random() >= 0.1) return;
  const rect = playRootRef.value.getBoundingClientRect();
  const width = Math.max(18, Math.floor(rect.width / (10 + Math.random() * 10)));
  const height = Math.max(8, Math.floor(width / 4));
  const x = Math.max(0, Math.floor(Math.random() * Math.max(1, rect.width - width)));
  const y = Math.max(0, Math.floor(Math.random() * Math.max(1, rect.height - height)));
  pushPageEffect({
    src: getMinigameAssetUrl('icesparkle.png'),
    x,
    y,
    width,
    height,
    durationMs: 1000,
  });
};

const syncSparkleTimer = () => {
  clearSparkleTimer();
  if (props.state?.gameId === 'ice-age') {
    sparkleTimer = window.setInterval(maybeSpawnIceSparkle, 880);
  }
};

watch(
  () => props.state?.gameId,
  () => {
    pageEffects.value = [];
    syncSparkleTimer();
  },
  { immediate: true }
);

watch(
  () => props.state?.animation?.pageEffects,
  (effects) => {
    (Array.isArray(effects) ? effects : []).forEach((effect) => {
      if (String(effect?.type || '') !== 'ice_sparkle' || !playRootRef.value) return;
      const rect = playRootRef.value.getBoundingClientRect();
      pushPageEffect({
        src: getMinigameAssetUrl('icesparkle.png'),
        x: Number(effect.x ?? rect.width * 0.5),
        y: Number(effect.y ?? rect.height * 0.5),
        width: Number(effect.width || 36),
        height: Number(effect.height || 9),
        durationMs: Number(effect.durationMs || 1000),
      });
    });
  }
);

onBeforeUnmount(() => {
  clearSparkleTimer();
});
</script>

<style scoped>
.minigame-title {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  overflow: hidden;
  font-size: calc(1.72rem * var(--ui-scale));
  max-width: 13.5ch;
  line-height: 1.04;
  padding-bottom: 0.08em;
}

.minigame-page-effects-layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 5;
}

.minigame-page-effect {
  position: absolute;
  object-fit: contain;
  user-select: none;
  -webkit-user-drag: none;
  animation: page-sparkle-fade calc(var(--page-effect-duration-ms) * 1ms) ease-out forwards;
}

@keyframes page-sparkle-fade {
  0% {
    opacity: 1;
    transform: scale(0.92);
  }
  100% {
    opacity: 0;
    transform: scale(1.06);
  }
}
</style>
