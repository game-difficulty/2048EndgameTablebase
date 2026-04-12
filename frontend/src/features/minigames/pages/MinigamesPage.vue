<template>
  <MinigameMenuPage
    v-if="currentView === 'menu'"
    :sections="menuSections"
    :difficulty="difficulty"
    :focus-game-id="lastMenuFocusGameId"
    :ws-status="wsStatus"
    :connection-badge-class="connectionBadgeClass"
    @set-difficulty="setDifficulty"
    @start-game="startGame"
  />
  <MinigamePlayPage
    v-else
    :state="gameState"
    :overlay="overlay"
    :toast-message="toastMessage"
    :ws-status="wsStatus"
    :connection-badge-class="connectionBadgeClass"
    @back-menu="backToMenu"
    @new-game="newGame"
    @info="requestInfo"
    @custom-action="triggerCustomAction"
    @use-powerup="usePowerup"
    @cancel-interaction="cancelInteraction"
    @cell-click="handleBoardCellClick"
    @close-overlay="closeOverlay"
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

const {
  wsStatus,
  connectionBadgeClass,
  menuSections,
  difficulty,
  currentView,
  gameState,
  lastMenuFocusGameId,
  toastMessage,
  overlay,
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
} = useMinigameSession(toRef(props, 'active'));
</script>
