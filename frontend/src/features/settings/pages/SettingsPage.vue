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
        v-for="tab in ['builder', 'game', 'theme']"
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
      <div v-show="activeSubTab === 'builder'" class="flex flex-col space-y-2 animate-fade-in">
        <div class="grid min-h-[360px] grid-cols-1 gap-6 items-stretch md:grid-cols-[minmax(0,1fr)_minmax(0,1.618fr)]">
          <div class="h-full space-y-5">
            <div class="flex flex-col">
              <label class="ui-control font-bold text-text-main mb-2 uppercase tracking-wider">{{ $t('settings.builder.category') }}</label>
              <UiSelect
                v-model="selectedCategory"
                class="w-full"
                :options="categoryOptions"
                :aria-label="$t('settings.builder.category')"
                trigger-class="w-full rounded-lg border border-border-main bg-bg-main px-3 py-2 ui-body font-bold text-text-main shadow-sm hover:border-accent/45"
                option-class="ui-body font-bold"
              />
            </div>

            <div class="flex flex-col">
              <label class="ui-control font-bold text-text-main mb-2 uppercase tracking-wider">{{ $t('settings.builder.pattern') }}</label>
              <UiSelect
                v-model="selectedPattern"
                class="w-full"
                :options="patternOptions"
                :aria-label="$t('settings.builder.pattern')"
                trigger-class="w-full rounded-lg border border-border-main bg-bg-main px-3 py-2 ui-body font-bold text-text-main shadow-sm hover:border-accent/45"
                option-class="ui-body font-bold"
              />
            </div>

            <div class="flex flex-col">
              <label class="ui-control font-bold text-text-main mb-2 uppercase tracking-wider">{{ $t('settings.builder.target') }}</label>
              <UiSelect
                v-model="selectedTarget"
                class="w-full"
                :options="targetOptions"
                :aria-label="$t('settings.builder.target')"
                trigger-class="w-full rounded-lg border border-border-main bg-bg-main px-3 py-2 ui-body font-bold text-text-main shadow-sm hover:border-accent/45"
                option-class="ui-body font-bold"
              />
            </div>

            <div class="flex flex-col border-t border-border-main pt-4 mt-2">
              <label class="ui-control font-bold text-text-main mb-2 uppercase tracking-wider">{{ $t('settings.builder.path') }}</label>
              <div class="flex gap-2 items-start">
                <textarea v-model="buildPath" rows="3" class="w-full bg-bg-main border border-border-main rounded-lg px-3 py-2 ui-kicker text-text-main outline-none focus:border-accent transition-colors flex-1 appearance-none resize-y" />
                <button @click="browseFolder" class="px-3 py-1.5 bg-btn-bg text-white rounded-lg font-black ui-control hover:bg-btn-hover active:scale-95 transition-all whitespace-nowrap shadow-sm">
                  {{ $t('settings.builder.browse') }}
                </button>
              </div>
              <div v-if="normalizedBuildPaths.length" class="mt-2 flex flex-col gap-1">
                <div v-for="(path, index) in normalizedBuildPaths" :key="`${index}-${path}`" class="flex min-w-0 items-center gap-2 text-xs text-text-muted">
                  <span class="shrink-0 rounded border border-border-main px-1.5 py-0.5 font-bold uppercase text-text-main">
                    {{ index === 0 ? $t('settings.builder.hotPath') : $t('settings.builder.coldPath') }}
                  </span>
                  <span class="truncate">{{ path }}</span>
                </div>
              </div>
            </div>
          </div>

          <div class="h-full bg-border-main/5 p-5 rounded-2xl border border-border-main shadow-inner">
            <div class="flex h-full flex-col gap-4">
              <div class="grid content-start grid-cols-1 gap-4 sm:grid-cols-2">
              <div class="flex flex-col">
                <label class="builder-field-label">
                  <span>{{ $t('settings.builder.algorithmMode') }}</span>
                  <span class="setting-tooltip" tabindex="0" :title="$t('settings.tooltips.algorithmMode')" :data-tooltip="$t('settings.tooltips.algorithmMode')">?</span>
                </label>
                <UiSelect
                  v-model="builderAlgorithm"
                  class="w-full"
                  :options="[
                    { value: 'classic', label: $t('settings.builder.algorithmClassic') },
                    { value: 'ad', label: $t('settings.builder.algorithmAD'), disabled: selectedPatternIsVariant },
                    { value: 'ex', label: $t('settings.builder.algorithmEX') },
                    { value: 'exad', label: $t('settings.builder.algorithmEXAD'), disabled: selectedPatternIsVariant },
                    { value: 'bc', label: $t('settings.builder.algorithmBC') },
                  ]"
                  :aria-label="$t('settings.builder.algorithmMode')"
                  trigger-class="w-full rounded-lg border border-border-main bg-bg-main px-3 py-2 ui-control font-black text-text-main shadow-sm hover:border-accent/45"
                  option-class="ui-control font-black"
                  @change="handleBuilderAlgorithmChange"
                />
              </div>

              <div class="flex flex-col">
                <label class="builder-field-label">
                  <span>{{ $t('settings.builder.successRateDtype') }}</span>
                  <span class="setting-tooltip" tabindex="0" :title="$t('settings.tooltips.successRateDtype')" :data-tooltip="$t('settings.tooltips.successRateDtype')">?</span>
                </label>
                <UiSelect
                  v-model="builderSuccessRateDtype"
                  class="w-full"
                  :options="successRateDtypeOptions"
                  :aria-label="$t('settings.builder.successRateDtype')"
                  trigger-class="w-full rounded-lg border border-border-main bg-bg-main px-3 py-2 ui-control font-black text-text-main shadow-sm hover:border-accent/45"
                  option-class="ui-control font-black"
                  @change="handleSuccessRateDtypeChange"
                />
              </div>

              <div v-if="builderUsesBC" class="flex flex-col">
                <label class="builder-field-label">
                  <span>{{ $t('settings.builder.bcFamilyModulus') }}</span>
                  <span class="setting-tooltip" tabindex="0" :title="$t('settings.tooltips.bcFamilyModulus')" :data-tooltip="$t('settings.tooltips.bcFamilyModulus')">?</span>
                </label>
                <input
                  type="number"
                  min="1"
                  max="65535"
                  step="1"
                  v-model.number="builderBCFamilyModulus"
                  :aria-label="$t('settings.builder.bcFamilyModulus')"
                  @change="handleBCFamilyModulusChange"
                  class="builder-number-input h-[42px] w-full bg-bg-main border border-border-main rounded-lg px-3 ui-control font-black text-text-main outline-none hover:border-accent transition-colors shadow-sm"
                />
              </div>

              <div class="flex flex-col">
                <label class="builder-field-label">
                  <span>{{ $t('settings.builder.deletionThreshold') }}</span>
                  <span class="setting-tooltip" tabindex="0" :title="$t('settings.tooltips.deletionThreshold')" :data-tooltip="$t('settings.tooltips.deletionThreshold')">?</span>
                </label>
                <div class="group relative min-w-0">
                  <input type="number" step="0.01" min="0" max="0.999999" :value="deletionThresholdInput" :disabled="builderDeletionThresholdMode === 'off'" :aria-label="$t('settings.builder.deletionThreshold')" @input="handleDeletionThresholdInput" @change="handleDeletionThresholdChange" class="builder-number-input h-[42px] w-full bg-bg-main border border-border-main rounded-lg px-3 pr-9 ui-control font-black text-text-main outline-none hover:border-accent transition-colors shadow-sm disabled:cursor-not-allowed disabled:opacity-50" />
                  <div class="pointer-events-none absolute inset-y-1 right-1 flex w-5 flex-col overflow-hidden rounded-md border border-border-main/70 bg-bg-card/90 opacity-0 shadow-sm transition-opacity duration-150 group-hover:pointer-events-auto group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:opacity-100" :class="{ 'hidden': builderDeletionThresholdMode === 'off' }">
                    <button type="button" @click="stepDeletionThreshold(1)" class="number-spin-btn border-b border-border-main/60" aria-label="Increase deletion threshold">
                      ▲
                    </button>
                    <button type="button" @click="stepDeletionThreshold(-1)" class="number-spin-btn" aria-label="Decrease deletion threshold">
                      ▼
                    </button>
                  </div>
                </div>
              </div>

              <div class="flex flex-col">
                <label class="builder-field-label">
                  <span>{{ $t('settings.builder.deletionThresholdModeShort') }}</span>
                  <span class="setting-tooltip" tabindex="0" :title="$t('settings.tooltips.deletionThresholdMode')" :data-tooltip="$t('settings.tooltips.deletionThresholdMode')">?</span>
                </label>
                <UiSelect
                  v-model="builderDeletionThresholdMode"
                  class="w-full"
                  :options="[
                    { value: 'off', label: $t('settings.builder.deletionThresholdOff') },
                    { value: 'absolute', label: $t('settings.builder.deletionThresholdAbsolute') },
                    { value: 'relative', label: $t('settings.builder.deletionThresholdRelative') },
                  ]"
                  :aria-label="$t('settings.builder.deletionThresholdMode')"
                  trigger-class="w-full rounded-lg border border-border-main bg-bg-main px-3 py-2 ui-control font-black text-text-main shadow-sm hover:border-accent/45"
                  option-class="ui-control font-black"
                  @change="handleDeletionThresholdModeChange"
                />
              </div>
            </div>

            <div class="grid grid-cols-1 gap-y-4 gap-x-5 border-t border-border-main pt-4 sm:grid-cols-3">
              <label class="builder-toggle-option cursor-pointer group">
                <div class="builder-toggle-switch relative">
                  <input type="checkbox" v-model="builderCompress" class="sr-only peer" @change="handleCompressChange" />
                  <div class="w-10 h-5 bg-border-main/30 rounded-full peer peer-checked:bg-accent transition-colors"></div>
                  <div class="absolute left-1 top-1 w-3 h-3 bg-white rounded-full transition-transform peer-checked:translate-x-5 shadow-sm"></div>
                </div>
                <span class="builder-toggle-copy">
                  <span class="builder-toggle-text ui-body font-bold text-text-main group-hover:text-accent transition-colors">{{ $t('settings.builder.compress') }}</span>
                  <span class="setting-tooltip builder-toggle-tooltip" tabindex="0" :title="$t('settings.tooltips.compress')" :data-tooltip="$t('settings.tooltips.compress')">?</span>
                </span>
              </label>

              <label class="builder-toggle-option cursor-pointer group">
                <div class="builder-toggle-switch relative">
                  <input type="checkbox" v-model="builderCompressTempFiles" class="sr-only peer" @change="handleCompressTempFilesChange" />
                  <div class="w-10 h-5 bg-border-main/30 rounded-full peer peer-checked:bg-accent transition-colors"></div>
                  <div class="absolute left-1 top-1 w-3 h-3 bg-white rounded-full transition-transform peer-checked:translate-x-5 shadow-sm"></div>
                </div>
                <span class="builder-toggle-copy">
                  <span class="builder-toggle-text ui-body font-bold text-text-main group-hover:text-accent transition-colors">{{ $t('settings.builder.compressTemp') }}</span>
                  <span class="setting-tooltip builder-toggle-tooltip" tabindex="0" :title="$t('settings.tooltips.compressTemp')" :data-tooltip="$t('settings.tooltips.compressTemp')">?</span>
                </span>
              </label>

              <label class="builder-toggle-option cursor-pointer group" v-if="builderSupportsOptimalOnly">
                <div class="builder-toggle-switch relative">
                  <input type="checkbox" v-model="builderOptimalBranchOnly" class="sr-only peer" @change="handleOptimalBranchOnlyChange" />
                  <div class="w-10 h-5 bg-border-main/30 rounded-full peer peer-checked:bg-accent transition-colors"></div>
                  <div class="absolute left-1 top-1 w-3 h-3 bg-white rounded-full transition-transform peer-checked:translate-x-5 shadow-sm"></div>
                </div>
                <span class="builder-toggle-copy">
                  <span class="builder-toggle-text ui-body font-bold text-text-main group-hover:text-accent transition-colors">{{ $t('settings.builder.optimalOnly') }}</span>
                  <span class="setting-tooltip builder-toggle-tooltip" tabindex="0" :title="$t('settings.tooltips.optimalOnly')" :data-tooltip="$t('settings.tooltips.optimalOnly')">?</span>
                </span>
              </label>

              <label class="builder-toggle-option cursor-pointer group" v-if="builderUsesAdvanced">
                <div class="builder-toggle-switch relative">
                  <input type="checkbox" v-model="builderChunkedSolve" class="sr-only peer" @change="handleChunkedSolveChange" />
                  <div class="w-10 h-5 bg-border-main/30 rounded-full peer peer-checked:bg-accent transition-colors"></div>
                  <div class="absolute left-1 top-1 w-3 h-3 bg-white rounded-full transition-transform peer-checked:translate-x-5 shadow-sm"></div>
                </div>
                <span class="builder-toggle-copy">
                  <span class="builder-toggle-text ui-body font-bold text-text-main group-hover:text-accent transition-colors">{{ $t('settings.builder.chunkedSolve') }}</span>
                  <span class="setting-tooltip builder-toggle-tooltip" tabindex="0" :title="$t('settings.tooltips.chunkedSolve')" :data-tooltip="$t('settings.tooltips.chunkedSolve')">?</span>
                </span>
              </label>
            </div>

            <div v-if="builderUsesAdvanced" class="pt-1">
              <div class="flex justify-between items-center mb-2">
                <label class="flex items-center gap-2 ui-control font-black text-text-main uppercase tracking-wider opacity-70">
                  <span>{{ $t('settings.builder.smallTileSumLimit') }}</span>
                  <span class="setting-tooltip" tabindex="0" :title="$t('settings.tooltips.smallTileSumLimit')" :data-tooltip="$t('settings.tooltips.smallTileSumLimit')">?</span>
                </label>
                <span class="pill-badge pill-badge-soft font-mono shadow-sm">{{ builderSmallTileSumLimit }}</span>
              </div>
              <input type="range" min="20" max="120" step="2" v-model.number="builderSmallTileSumLimit" @change="handleSmallTileSumLimitChange" class="w-full accent-accent cursor-pointer" />
            </div>
            </div>
          </div>
        </div>

        <div class="border-t border-border-main pt-6">
          <button
            @click="startBuild"
            :disabled="isBuilding"
            :class="[
              'relative isolate w-full overflow-hidden rounded-xl py-4 shadow-lg transition-all active:scale-[0.98]',
              isBuilding
                ? 'cursor-not-allowed border border-border-main/70 bg-btn-bg/45 text-white'
                : 'btn-prominent text-white hover:shadow-xl hover:-translate-y-0.5'
            ]"
          >
            <span v-if="isBuilding" class="absolute inset-0 bg-border-main/15" />
            <span
              v-if="isBuilding"
              class="absolute inset-y-0 left-0 rounded-[inherit] bg-gradient-to-r from-accent/45 via-accent/70 to-accent transition-[width] duration-300 ease-out"
              :style="{ width: `${buildProgressPercent}%` }"
            />
            <span class="relative z-10 flex items-center justify-center gap-4 px-4">
              <span class="text-xl font-black tracking-widest">
                {{ isBuilding ? $t('settings.builder.progress') : $t('settings.builder.build') }}
              </span>
              <span
                v-if="isBuilding"
                class="rounded-full border border-white/15 bg-black/10 px-2.5 py-1 ui-body font-black tracking-wide text-white/90 shadow-sm backdrop-blur-sm"
              >
                {{ buildProgressDisplay }}
              </span>
            </span>
          </button>
        </div>
      </div>

      <div v-show="activeSubTab === 'game'" class="space-y-8 animate-fade-in max-w-2xl px-2">
        <div class="flex flex-col w-full pb-4 border-b border-border-main">
          <label class="ui-control font-bold text-text-main mb-4 uppercase tracking-wider">{{ $t('settings.game.language') }}</label>
          <div class="flex gap-2">
            <button v-for="lang in ['en', 'zh']" :key="lang" @click="changeLanguage(lang)"
              :class="['flex-1 py-3 px-4 rounded-xl font-bold transition-all flex items-center justify-center gap-2 border-2', config.language === lang ? 'surface-prominent text-white shadow-md scale-[1.02]' : 'bg-bg-main text-text-main border-border-main hover:border-accent/60']">
              <i :class="lang === 'zh' ? 'fas fa-language' : 'fas fa-globe-americas'"></i>
              {{ lang === 'zh' ? '简体中文' : 'English' }}
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
            <button v-for="t in themes" :key="t" @click="setTheme(t)"
              :class="['px-3 py-1.5 rounded-md ui-control font-bold transition-all border', (config.theme === t && !config.use_custom_theme) ? 'surface-prominent text-white shadow-md' : 'bg-bg-main text-text-main border-border-main hover:border-accent/40']">
              {{ t }}
            </button>

            <button @click="setCustomMode"
              :class="['px-3 py-1.5 rounded-md ui-control font-bold transition-all border', config.use_custom_theme ? 'surface-prominent text-white shadow-md' : 'bg-bg-main text-text-main border-border-main hover:border-accent/40']">
              {{ $t('settings.theme.custom_label') || 'Custom' }}
            </button>
          </div>

          <div v-if="currentPalette.length > 0" class="flex w-full h-8 rounded-lg overflow-hidden border border-border-main shadow-sm transition-all duration-500 mb-2">
            <div v-for="(color, i) in currentPalette" :key="i"
              class="flex-1 h-full animate-grow-x"
              :style="{ backgroundColor: color, transitionDelay: `${i * 30}ms` }"
              :title="`Tile ${2**(i+1)}: ${color}`">
            </div>
          </div>
        </div>

        <div class="border-t border-border-main pt-6">
          <label class="ui-control font-bold text-text-main mb-4 block uppercase tracking-wider">{{ $t('settings.theme.colors') }}</label>
          <div class="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-3">
            <div v-for="index in 16" :key="index" class="flex flex-col items-center gap-1">
              <span class="ui-kicker font-bold text-text-main">{{ 2**index }}</span>
              <div class="relative w-10 h-10 rounded-lg shadow-sm border border-white/20 overflow-hidden group">
                <input type="color" v-model="config.custom_colors[index-1]" @change="saveCustomColor"
                  class="absolute -top-1 -left-1 w-12 h-12 cursor-pointer border-none p-0 bg-transparent" />
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
import { computed, toRef } from 'vue';

import UiSelect from '../../../components/UiSelect.vue';
import { useSettingsSession } from '../composables/useSettingsSession';

const props = defineProps({
  active: {
    type: Boolean,
    default: false,
  },
});

  const {
    activeSubTab,
    settingsLoaded,
    wsStatus,
    categories,
  targetTiles,
  config,
  themes,
  currentPalette,
  selectedCategory,
  selectedPattern,
  selectedTarget,
  selectedPatternIsVariant,
  buildPath,
  normalizedBuildPaths,
  isBuilding,
  builderAlgorithm,
  builderCompress,
  builderCompressTempFiles,
  builderOptimalBranchOnly,
  builderChunkedSolve,
  builderBCFamilyModulus,
  builderSuccessRateDtype,
  builderSmallTileSumLimit,
  builderDeletionThresholdMode,
  deletionThresholdInput,
  filteredPatterns,
  buildProgressPercent,
  buildProgressDisplay,
  saveSetting,
  handleBuilderAlgorithmChange,
  handleCompressChange,
  handleCompressTempFilesChange,
  handleOptimalBranchOnlyChange,
  handleChunkedSolveChange,
  handleBCFamilyModulusChange,
  handleSuccessRateDtypeChange,
  handleDeletionThresholdModeChange,
  handleDeletionThresholdInput,
  handleDeletionThresholdChange,
  stepDeletionThreshold,
  handleSmallTileSumLimitChange,
  saveCustomColor,
  setTheme,
  changeLanguage,
  setCustomMode,
  browseFolder,
  startBuild,
} = useSettingsSession(toRef(props, 'active'));

const categoryOptions = computed(() =>
  Object.keys(categories.value || {}).map((category) => ({
    value: category,
    label: category,
  }))
);

const patternOptions = computed(() =>
  (filteredPatterns.value || []).map((pattern) => ({
    value: pattern,
    label: pattern,
  }))
);

const targetOptions = computed(() =>
  (targetTiles.value || []).map((target) => ({
    value: target,
    label: target,
  }))
);

const successRateDtypeOptions = [
  { value: 'uint32', label: 'uint32' },
  { value: 'uint64', label: 'uint64' },
  { value: 'float32', label: 'float32' },
  { value: 'float64', label: 'float64' },
  { value: '1-float32', label: '1-float32' },
  { value: '1-float64', label: '1-float64' },
];

const builderUsesAdvanced = computed(() =>
  builderAlgorithm.value === 'ad' || builderAlgorithm.value === 'exad'
);

const builderUsesBC = computed(() => builderAlgorithm.value === 'bc');

const builderSupportsOptimalOnly = computed(() =>
  !builderUsesAdvanced.value && !builderUsesBC.value
);
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

.builder-toggle-option {
  display: grid;
  min-height: 3.25rem;
  grid-template-columns: auto minmax(0, 1fr);
  align-items: center;
  column-gap: 0.75rem;
}

.builder-toggle-switch {
  align-self: center;
  margin-top: 0;
}

.builder-toggle-copy {
  display: flex;
  min-width: 0;
  min-height: 3.25rem;
  align-items: center;
  column-gap: 0.5rem;
}

.builder-toggle-text {
  min-width: 0;
  flex: 0 1 auto;
  line-height: 1.3;
}

.builder-toggle-tooltip {
  flex: 0 0 auto;
}

.builder-field-label {
  display: flex;
  min-height: 1.65rem;
  align-items: flex-start;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
  color: var(--text-main);
  font-size: var(--font-ui-sm);
  font-weight: 900;
  line-height: 1.2;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  opacity: 0.7;
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

.builder-number-input {
  appearance: textfield;
  -moz-appearance: textfield;
}

.builder-number-input::-webkit-inner-spin-button,
.builder-number-input::-webkit-outer-spin-button {
  -webkit-appearance: none;
  margin: 0;
}

.number-spin-btn {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  color: var(--text-secondary);
  font-size: 8px;
  font-weight: 900;
  line-height: 1;
  transition: background-color 0.15s ease, color 0.15s ease;
}

.number-spin-btn:hover {
  background: var(--btn-bg);
  color: #fff;
}

.setting-tooltip {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1rem;
  height: 1rem;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--border-main) 88%, transparent);
  background: color-mix(in srgb, var(--bg-main) 78%, var(--bg-card) 22%);
  color: var(--text-secondary);
  font-size: 0.68rem;
  font-weight: 900;
  line-height: 1;
  cursor: help;
  transition: all 0.18s ease;
}

.setting-tooltip:hover,
.setting-tooltip:focus-visible {
  color: var(--text-main);
  border-color: color-mix(in srgb, var(--accent) 46%, var(--border-main) 54%);
  background: color-mix(in srgb, var(--accent) 16%, var(--bg-card) 84%);
  outline: none;
}

.setting-tooltip::after {
  content: attr(data-tooltip);
  position: absolute;
  left: 50%;
  bottom: calc(100% + 10px);
  transform: translateX(-50%) translateY(4px);
  min-width: 220px;
  max-width: 320px;
  padding: 0.6rem 0.7rem;
  border-radius: 0.75rem;
  background: color-mix(in srgb, var(--bg-main) 92%, var(--text-main) 8%);
  border: 1px solid color-mix(in srgb, var(--border-main) 82%, transparent);
  color: var(--text-main);
  font-size: var(--font-ui-caption);
  font-weight: 700;
  line-height: 1.35;
  text-transform: none;
  letter-spacing: normal;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16);
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.18s ease, transform 0.18s ease;
  z-index: 30;
  white-space: normal;
}

.setting-tooltip:hover::after,
.setting-tooltip:focus-visible::after {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
}
</style>
