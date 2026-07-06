<template>
  <div class="page-root pt-6">
    <div class="w-full max-w-lg flex flex-col items-center">
      <div class="flex justify-between w-full mb-6 items-center">
        <div class="flex items-center gap-4">
          <h1 class="text-6xl font-bold text-text-main leading-none">2048</h1>
          <div class="flex flex-col gap-1.5">
            <span :class="['badge-base', wsStatus === 'connected' ? 'badge-connection-connected' : 'badge-connection-pending']">
              {{ $t(`status.${wsStatus.toLowerCase().replace('...', '')}`) }}
            </span>
            <span class="badge-base badge-link inline-flex items-center justify-center px-2.5 py-1">
              {{ aiWorkerReady ? 'WASM' : 'WASM...' }}
            </span>
          </div>
        </div>
        <div class="flex space-x-2">
          <div class="bg-board-bg w-[122px] h-[56px] flex flex-col items-center justify-center rounded-md relative shadow-sm transition-all duration-300">
            <span class="text-text-secondary ui-caption font-black uppercase leading-none mb-1 tracking-tight">{{ $t('labels.score') }}</span>
            <span class="font-black text-white leading-none tabular-nums" style="font-size: calc(22px * var(--ui-scale));">{{ score.current }}</span>

            <transition-group name="float-up" tag="div" class="absolute inset-x-0 bottom-4 pointer-events-none flex justify-center">
              <div v-for="anim in scoreAnimations" :key="anim.id" class="absolute ui-metric text-[#3EB489] font-black z-50 float-anim drop-shadow-[0_0_5px_rgba(62,180,137,0.4)]">
                +{{ anim.value }}
              </div>
            </transition-group>
          </div>
          <div class="bg-board-bg w-[122px] h-[56px] flex flex-col items-center justify-center rounded-md shadow-sm transition-all duration-300">
            <span class="text-text-secondary ui-caption font-black uppercase leading-none mb-1 tracking-tight">{{ $t('labels.best') }}</span>
            <span class="font-black text-white leading-none tabular-nums" style="font-size: calc(22px * var(--ui-scale));">{{ score.best }}</span>
          </div>
        </div>
      </div>

      <div class="w-full mb-4">
        <div class="flex w-full space-x-2">
          <button @click="triggerAction('INIT_GAME')" class="flex-1 bg-btn-bg text-white font-bold py-2 px-2 rounded hover:bg-btn-hover ui-body whitespace-nowrap">
            {{ $t('buttons.newGame') }}
          </button>
          <button @click="triggerAction('UNDO')" class="flex-1 bg-btn-bg text-white font-bold py-2 px-2 rounded hover:bg-btn-hover ui-body whitespace-nowrap">
            {{ $t('buttons.undo') }}
          </button>
          <button
            @click="triggerAction('AI_STEP')"
            :disabled="!aiWorkerReady"
            :class="!aiWorkerReady ? 'opacity-55 cursor-not-allowed' : ''"
            class="flex-1 bg-btn-bg text-white font-bold py-2 px-2 rounded hover:bg-btn-hover ui-body whitespace-nowrap"
          >
            {{ $t('buttons.oneStep') }}
          </button>
          <button
            @click="toggleAI"
            :disabled="!aiWorkerReady"
            :class="[aiEnabled ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600', !aiWorkerReady ? 'opacity-55 cursor-not-allowed' : '']"
            class="flex-1 text-white font-bold py-2 px-2 rounded ui-body whitespace-nowrap"
          >
            {{ aiEnabled ? $t('buttons.aiOn') : $t('buttons.aiOff') }}
          </button>
        </div>
      </div>

      <div class="w-full flex space-x-2 mb-4">
        <input type="text" v-model="hexInput" :placeholder="$t('inputs.hexPlaceholder')" class="flex-1 px-3 py-2 bg-bg-main border border-border-main rounded ui-body text-text-main font-mono tracking-widest outline-none focus:border-accent transition-all" />
        <button @click="setBoard" class="bg-btn-bg text-white font-bold py-1 px-5 rounded hover:bg-btn-hover ui-body shadow-sm transition-all active:scale-95">
          {{ $t('buttons.set') }}
        </button>
        <button
          type="button"
          @click="writeCurrentBoardToHex"
          class="bg-btn-bg text-white font-bold py-1 px-4 rounded hover:bg-btn-hover ui-body shadow-sm transition-all active:scale-95 whitespace-nowrap"
          :title="$t('buttons.loadCurrentBoard')"
        >
          {{ $t('buttons.loadCurrentBoard') }}
        </button>
      </div>

      <div ref="boardHotkeyTarget" tabindex="-1" class="w-full outline-none focus:outline-none">
        <BaseBoard :board="board" :metadata="metadata" @swipe="handleBoardSwipe" />
      </div>

      <div class="w-full mt-6 bg-ctrl-bg rounded-md p-4 flex flex-col space-y-4 shadow-sm">
        <div class="w-full flex justify-between gap-4 mb-4">
          <div class="flex-1 flex flex-col justify-center">
            <span class="text-text-main ui-body font-bold opacity-80 mb-1">{{ $t('labels.gameDifficulty') }}</span>
            <input type="range" class="w-full" :min="0" :max="100" :step="1" v-model.number="difficulty" @change="handleUpdateSettings($event)" />
          </div>
          <div class="flex-1 flex flex-col justify-center">
            <span class="text-text-main ui-body font-bold opacity-80 mb-1">{{ $t('labels.aiSpeed') }}</span>
            <input type="range" class="w-full" :min="0" :max="200" :step="1" v-model.number="aiSpeed" @change="handleUpdateSettings($event)" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, toRef } from 'vue';

import BaseBoard from '../../../components/BaseBoard.vue';
import { refocusBoardHotkeyTarget } from '../../../utils/boardHotkeyFocus';
import { useGamerSession } from '../composables/useGamerSession';

const props = defineProps({
  active: { type: Boolean, default: true },
});

const boardHotkeyTarget = ref(null);

const {
  board,
  metadata,
  score,
  wsStatus,
  aiEnabled,
  difficulty,
  aiSpeed,
  hexInput,
  scoreAnimations,
  aiWorkerReady,
  triggerAction,
  toggleAI,
  updateSettings,
  setBoard,
  writeCurrentBoardToHex,
} = useGamerSession(toRef(props, 'active'));

const handleUpdateSettings = (event) => {
  updateSettings();
  refocusBoardHotkeyTarget(boardHotkeyTarget, event?.target);
};

const handleBoardSwipe = (direction) => {
  triggerAction('USER_MOVE', { dir: direction });
};
</script>
