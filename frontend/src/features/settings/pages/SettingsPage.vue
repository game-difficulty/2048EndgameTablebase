<template>
  <div class="page-root pt-6">
    <div class="w-full max-w-4xl flex flex-col items-center">
      <div class="w-full flex items-center justify-between mb-8">
        <h1 class="text-4xl font-extrabold text-text-main tracking-tight">{{ $t('settings.title') }}</h1>
        <div :class="['badge-base shadow-sm transition-colors', wsStatus === 'connected' ? 'badge-connection-connected' : 'badge-connection-pending']">
          {{ wsStatus === 'connected' ? $t('status.connected') : $t('status.connecting') }}
        </div>
      </div>

      <div class="w-full flex p-1 bg-border-main/20 rounded-xl mb-6 backdrop-blur-sm self-start max-w-md">
        <button
          v-for="tab in ['game', 'theme']"
          :key="tab"
          @click="activeSubTab = tab"
          :class="['flex-1 py-2 ui-body font-bold rounded-lg border border-transparent transition-all duration-300', activeSubTab === tab ? 'surface-prominent text-white shadow-sm' : 'text-text-secondary hover:text-text-main']"
        >
          {{ $t(`settings.tabs.${tab}`) }}
        </button>
      </div>

      <div class="w-full bg-bg-card border border-border-main rounded-2xl p-6 shadow-xl min-h-[400px]">
        <div v-if="!settingsLoaded" class="flex min-h-[340px] items-center justify-center">
          <div class="ui-body font-bold text-text-secondary">{{ $t('settings.loading') }}</div>
        </div>
        <template v-else>
          <div v-show="activeSubTab === 'game'" class="space-y-8 animate-fade-in max-w-2xl px-2">
            <div class="flex flex-col w-full pb-4 border-b border-border-main">
              <label class="ui-control font-bold text-text-main mb-4 uppercase tracking-wider">{{ $t('settings.game.language') }}</label>
              <div class="flex gap-2">
                <button
                  v-for="option in languageOptions"
                  :key="option.value"
                  @click="changeLanguage(option.value)"
                  :class="['flex-1 py-3 px-4 rounded-xl font-bold transition-all flex items-center justify-center gap-2 border-2', config.language === option.value ? 'surface-prominent text-white shadow-md scale-[1.02]' : 'bg-bg-main text-text-main border-border-main hover:border-accent/60']"
                >
                  <i :class="option.value === 'zh' ? 'fas fa-language' : 'fas fa-globe-americas'"></i>
                  {{ option.label }}
                </button>
              </div>
            </div>

            <div class="flex flex-col w-full">
              <div class="flex justify-between mb-2">
                <label class="ui-body font-bold text-text-main">{{ $t('settings.game.demoSpeed') }}</label>
                <span class="ui-body font-mono font-bold text-accent">{{ config.demo_speed }} ms</span>
              </div>
              <input type="range" class="w-full accent-accent" min="1" max="3000" step="1" v-model.number="config.demo_speed" @change="saveSetting('demo_speed')" />
            </div>

            <div class="flex flex-col w-full">
              <div class="flex justify-between mb-2">
                <label class="ui-body font-bold text-text-main">{{ $t('settings.game.spawnRate') }}</label>
                <span class="ui-body font-mono font-bold text-accent">{{ (config['4_spawn_rate'] * 100).toFixed(1) }}%</span>
              </div>
              <input type="range" class="w-full accent-accent" min="0" max="1" step="0.01" v-model.number="config['4_spawn_rate']" @change="saveSetting('4_spawn_rate')" />
            </div>

            <div class="flex flex-col w-full">
              <div class="flex justify-between mb-2">
                <label class="ui-body font-bold text-text-main">{{ $t('settings.game.fontSize') }}</label>
                <span class="ui-body font-mono font-bold text-accent">{{ config.font_size_factor }}%</span>
              </div>
              <input type="range" class="w-full accent-accent" min="50" max="150" step="5" v-model.number="config.font_size_factor" @change="saveSetting('font_size_factor')" />
            </div>

            <div class="flex flex-col w-full">
              <div class="flex justify-between mb-2">
                <label class="ui-body font-bold text-text-main">{{ $t('settings.game.uiScale') }}</label>
                <span class="ui-body font-mono font-bold text-accent">{{ config.ui_scale }}%</span>
              </div>
              <input type="range" class="w-full accent-accent" min="90" max="125" step="5" v-model.number="config.ui_scale" @change="saveSetting('ui_scale')" />
            </div>

            <div class="grid grid-cols-2 gap-4">
              <div class="flex min-w-0 items-center justify-between rounded-xl border border-border-main bg-bg-card p-4">
                <span class="font-bold text-text-main">{{ $t('settings.game.darkMode') }}</span>
                <label class="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" v-model="config.dark_mode" class="sr-only peer" @change="saveSetting('dark_mode')" />
                  <div class="w-11 h-6 bg-border-main/30 border border-border-main rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent shadow-inner"></div>
                </label>
              </div>

              <div class="flex min-w-0 items-center justify-between rounded-xl border border-border-main bg-bg-card p-4">
                <span class="font-bold text-text-main">{{ $t('settings.game.animation') }}</span>
                <label class="relative inline-flex items-center cursor-pointer">
                  <input type="checkbox" v-model="config.do_animation" class="sr-only peer" @change="saveSetting('do_animation')" />
                  <div class="w-11 h-6 bg-border-main/30 border border-border-main rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent shadow-inner"></div>
                </label>
              </div>
            </div>
          </div>

          <div v-show="activeSubTab === 'theme'" class="space-y-6 animate-fade-in">
            <div class="flex flex-col max-w-full">
              <label class="ui-control font-bold text-text-main mb-2 uppercase tracking-wider">{{ $t('settings.theme.theme') }}</label>
              <div class="flex flex-wrap gap-2 mb-4">
                <button
                  v-for="themeName in themes"
                  :key="themeName"
                  @click="setTheme(themeName)"
                  :class="['px-3 py-1.5 rounded-md ui-control font-bold transition-all border', (config.theme === themeName && !config.use_custom_theme) ? 'surface-prominent text-white shadow-md' : 'bg-bg-main text-text-main border-border-main hover:border-accent/40']"
                >
                  {{ themeName }}
                </button>

                <button
                  @click="setCustomMode"
                  :class="['px-3 py-1.5 rounded-md ui-control font-bold transition-all border', config.use_custom_theme ? 'surface-prominent text-white shadow-md' : 'bg-bg-main text-text-main border-border-main hover:border-accent/40']"
                >
                  {{ $t('settings.theme.custom_label') || 'Custom' }}
                </button>
              </div>

              <div v-if="currentPalette.length > 0" class="flex w-full h-8 rounded-lg overflow-hidden border border-border-main shadow-sm transition-all duration-500 mb-2">
                <div
                  v-for="(color, index) in currentPalette"
                  :key="index"
                  class="flex-1 h-full animate-grow-x"
                  :style="{ backgroundColor: color, transitionDelay: `${index * 30}ms` }"
                  :title="`Tile ${2 ** (index + 1)}: ${color}`"
                ></div>
              </div>
            </div>

            <div class="border-t border-border-main pt-6">
              <label class="ui-control font-bold text-text-main mb-4 block uppercase tracking-wider">{{ $t('settings.theme.colors') }}</label>
              <div class="grid grid-cols-8 gap-3">
                <div v-for="index in 16" :key="index" class="flex flex-col items-center gap-1">
                  <span class="ui-kicker font-bold text-text-main">{{ 2 ** index }}</span>
                  <div class="relative w-10 h-10 rounded-lg shadow-sm border border-white/20 overflow-hidden group">
                    <input
                      type="color"
                      v-model="config.custom_colors[index - 1]"
                      @change="saveCustomColor"
                      class="absolute -top-1 -left-1 w-12 h-12 cursor-pointer border-none p-0 bg-transparent"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup>
import { toRef } from 'vue';

import { useSettingsSession } from '../composables/useSettingsSession';

const props = defineProps({
  active: {
    type: Boolean,
    default: false,
  },
});

const languageOptions = [
  { value: 'en', label: 'English' },
  { value: 'zh', label: 'Chinese' },
];

const {
  activeSubTab,
  settingsLoaded,
  wsStatus,
  config,
  themes,
  currentPalette,
  saveSetting,
  saveCustomColor,
  setTheme,
  changeLanguage,
  setCustomMode,
} = useSettingsSession(toRef(props, 'active'));
</script>

<style scoped>
.animate-fade-in {
  animation: fadeIn 0.4s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: translateY(0); }
}

.animate-grow-x {
  animation: growX 0.6s ease-out forwards;
  transform-origin: left;
}

@keyframes growX {
  from { transform: scaleX(0); }
  to { transform: scaleX(1); }
}

input[type="color"]::-webkit-color-swatch-wrapper {
  padding: 0;
}

input[type="color"]::-webkit-color-swatch {
  border: none;
}

input[type="color"] {
  -webkit-appearance: none;
  appearance: none;
  border: none;
  background: none;
}
</style>
