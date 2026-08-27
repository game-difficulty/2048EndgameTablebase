<template>
  <MinigameMenuPage
    v-if="currentView === 'menu'"
    :sections="menuSections"
    :difficulty="difficulty"
    :focus-game-id="lastMenuFocusGameId"
    @set-difficulty="setDifficulty"
    @start-game="startGame"
  />
  <MinigamePlayPage
    v-else
    :state="gameState"
    :overlay="overlay"
    :toast-message="toastMessage"
    :ranked-status="rankedStatus"
    :active="active"
    @back-menu="backToMenu"
    @new-game="newGame"
    @info="requestInfo"
    @custom-action="triggerCustomAction"
    @use-powerup="usePowerup"
    @cancel-interaction="cancelInteraction"
    @cell-click="handleBoardCellClick"
    @swipe="move"
    @close-overlay="closeOverlay"
    @navigate-tab="forwardNavigateTab"
  />
</template>

<script setup>
import { toRef } from 'vue';

import { useMinigameSession } from '../composables/useMinigameSession';
import MinigameMenuPage from './MinigameMenuPage.vue';
import MinigamePlayPage from './MinigamePlayPage.vue';

const props = defineProps({
  active: { type: Boolean, default: false },
});
const emit = defineEmits(['navigate-tab']);
const forwardNavigateTab = (tabId, detail) => emit('navigate-tab', tabId, detail);

const {
  menuSections,
  difficulty,
  currentView,
  gameState,
  lastMenuFocusGameId,
  toastMessage,
  overlay,
  rankedStatus,
  closeOverlay,
  setDifficulty,
  startGame,
  backToMenu,
  newGame,
  requestInfo,
  triggerCustomAction,
  usePowerup,
  cancelInteraction,
  handleBoardCellClick,
  move,
} = useMinigameSession(toRef(props, 'active'));
</script>
