<template>
  <div class="page-root pt-6">
    <div class="w-full max-w-lg flex flex-col items-center">
      <div class="flex justify-between w-full mb-6 items-center">
        <div class="flex items-center gap-4">
          <button
            type="button"
            @click="openBrowserAi"
            class="group flex items-center gap-2 rounded-xl px-1 py-1 text-left transition-all hover:bg-bg-card/50"
            :title="$t('gamer.browserAi.title')"
          >
            <h1 class="text-6xl font-bold text-text-main leading-none transition-colors group-hover:text-accent">2048</h1>
          </button>
          <div class="flex flex-col gap-1.5">
            <span :class="['badge-base', wsStatus === 'connected' ? 'badge-connection-connected' : 'badge-connection-pending']">
              {{ $t(`status.${wsStatus.toLowerCase().replace('...', '')}`) }}
            </span>
            <button
              type="button"
              @click="openBrowserAi"
              class="badge-base badge-link inline-flex items-center justify-center px-2.5 py-1 transition-all"
              :title="$t('gamer.browserAi.title')"
            >
              <svg class="h-3.5 w-3.5" viewBox="0 0 16 16" aria-hidden="true">
                <path fill="currentColor" d="M9.75 2.5h3.75v3.75h-1.5V5.06L7.53 9.53 6.47 8.47 10.94 4H9.75z"/>
                <path fill="currentColor" d="M3.5 4.25A1.75 1.75 0 0 1 5.25 2.5H8v1.5H5.25a.25.25 0 0 0-.25.25v6.5c0 .138.112.25.25.25h6.5a.25.25 0 0 0 .25-.25V8h1.5v2.75a1.75 1.75 0 0 1-1.75 1.75h-6.5A1.75 1.75 0 0 1 3.5 10.75z"/>
              </svg>
            </button>
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
          <button @click="triggerAction('AI_STEP')" class="flex-1 bg-btn-bg text-white font-bold py-2 px-2 rounded hover:bg-btn-hover ui-body whitespace-nowrap">
            {{ $t('buttons.oneStep') }}
          </button>
          <button @click="toggleAI" :class="aiEnabled ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'" class="flex-1 text-white font-bold py-2 px-2 rounded ui-body whitespace-nowrap">
            {{ aiEnabled ? $t('buttons.aiOn') : $t('buttons.aiOff') }}
          </button>
        </div>
      </div>

      <div class="w-full flex space-x-2 mb-4">
        <input type="text" v-model="hexInput" :placeholder="$t('inputs.hexPlaceholder')" class="flex-1 px-3 py-2 bg-bg-main border border-border-main rounded ui-body text-text-main font-mono tracking-widest outline-none focus:border-accent transition-all" />
        <button @click="setBoard" class="bg-btn-bg text-white font-bold py-1 px-5 rounded hover:bg-btn-hover ui-body shadow-sm transition-all active:scale-95">
          {{ $t('buttons.set') }}
        </button>
      </div>

      <div class="w-full">
        <BaseBoard :board="board" :metadata="metadata" :dis32k="dis32k" />
      </div>

      <div class="w-full mt-6 bg-ctrl-bg rounded-md p-4 flex flex-col space-y-4 shadow-sm">
        <div class="w-full flex justify-between gap-4 mb-4">
          <div class="flex-1 flex flex-col justify-center">
            <span class="text-text-main ui-body font-bold opacity-80 mb-1">{{ $t('labels.gameDifficulty') }}</span>
            <input type="range" class="w-full" :min="0" :max="100" :step="1" v-model.number="difficulty" @change="updateSettings" />
          </div>
          <div class="flex-1 flex flex-col justify-center">
            <span class="text-text-main ui-body font-bold opacity-80 mb-1">{{ $t('labels.aiSpeed') }}</span>
            <input type="range" class="w-full" :min="0" :max="200" :step="1" v-model.number="aiSpeed" @change="updateSettings" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { toRef } from 'vue';

import BaseBoard from '../../../components/BaseBoard.vue';
import { useGamerSession } from '../composables/useGamerSession';

const props = defineProps({
  active: { type: Boolean, default: true },
});

const {
  board,
  metadata,
  score,
  wsStatus,
  aiEnabled,
  difficulty,
  aiSpeed,
  dis32k,
  hexInput,
  scoreAnimations,
  triggerAction,
  toggleAI,
  updateSettings,
  setBoard,
  openBrowserAi,
} = useGamerSession(toRef(props, 'active'));
</script>
