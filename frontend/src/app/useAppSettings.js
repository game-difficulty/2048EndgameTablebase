import { computed, ref } from 'vue';

import i18n from './i18n';
import { createWsClient } from '../services/ws/createWsClient';
import { applyTileColors } from '../utils/tileColors';

const EMPTY_COLOR_SET = Array(36).fill('#000000');

const DEFAULT_CONFIG = {
  colors: [...EMPTY_COLOR_SET],
  custom_colors: [...EMPTY_COLOR_SET],
  use_custom_theme: false,
  compress: false,
  optimal_branch_only: false,
  compress_temp_files: false,
  advanced_algo: false,
  chunked_solve: false,
  deletion_threshold: 0,
  SmallTileSumLimit: 96,
  success_rate_dtype: 'uint32',
  demo_speed: 40,
  record_player_slider_threshold: 1,
  dis_32k: false,
  dark_mode: false,
  '4_spawn_rate': 0.1,
  do_animation: true,
  font_size_factor: 100,
  ui_scale: 100,
  theme: 'Default',
  language: 'en',
};

const wsStatus = ref('connecting');
const loaded = ref(false);
const config = ref({ ...DEFAULT_CONFIG });
const categories = ref({});
const themeMap = ref({});
const targetTiles = ref([]);

let client = null;
let started = false;
const clientId = `app_settings_${Math.random().toString(36).slice(2, 9)}`;

const clonePalette = (palette, fallback = EMPTY_COLOR_SET) =>
  Array.isArray(palette) && palette.length > 0 ? [...palette] : [...fallback];

const mergeConfig = (nextConfig = {}) => {
  const previous = config.value;
  config.value = {
    ...previous,
    ...nextConfig,
    colors: clonePalette(nextConfig.colors ?? previous.colors, DEFAULT_CONFIG.colors),
    custom_colors: clonePalette(nextConfig.custom_colors ?? previous.custom_colors, DEFAULT_CONFIG.custom_colors),
  };
};

const setGlobalLocale = (language) => {
  if (!language) {
    return;
  }
  const globalLocale = i18n.global.locale;
  if (globalLocale && typeof globalLocale === 'object' && 'value' in globalLocale) {
    globalLocale.value = language;
    return;
  }
  i18n.global.locale = language;
};

const getResolvedPalette = () => {
  if (config.value.use_custom_theme && config.value.custom_colors?.length) {
    return config.value.custom_colors;
  }
  const themePalette = themeMap.value?.[config.value.theme];
  if (Array.isArray(themePalette) && themePalette.length > 0) {
    return themePalette;
  }
  if (config.value.colors?.length) {
    return config.value.colors;
  }
  return [];
};

const applyGlobalConfig = () => {
  setGlobalLocale(config.value.language || 'en');

  if (config.value.dark_mode) {
    document.documentElement.setAttribute('data-theme', 'dark');
  } else {
    document.documentElement.removeAttribute('data-theme');
  }

  const palette = getResolvedPalette();
  if (palette.length > 0) {
    applyTileColors(palette);
  }

  const fontScale = Number(config.value.font_size_factor) || DEFAULT_CONFIG.font_size_factor;
  document.documentElement.style.setProperty('--tile-font-scale', String(fontScale / 100));

  const uiScale = Number(config.value.ui_scale) || DEFAULT_CONFIG.ui_scale;
  document.documentElement.style.setProperty('--ui-scale', String(uiScale / 100));
};

const handleSettingsData = (payload = {}) => {
  categories.value = payload.categories || {};
  themeMap.value = payload.theme_map || {};
  targetTiles.value = payload.target_tiles || [];
  mergeConfig(payload.config || {});
  config.value.ui_scale = Number(payload?.config?.ui_scale ?? DEFAULT_CONFIG.ui_scale) || DEFAULT_CONFIG.ui_scale;
  loaded.value = true;
  applyGlobalConfig();
};

const handleSettingUpdated = (payload = {}) => {
  const { key, value } = payload;
  if (!key) {
    return;
  }

  if (key === 'colors') {
    mergeConfig({ colors: value });
  } else if (key === 'custom_colors') {
    mergeConfig({ custom_colors: value });
  } else if (key === 'ui_scale') {
    mergeConfig({ ui_scale: value });
  } else {
    mergeConfig({ [key]: value });
  }

  loaded.value = true;
  applyGlobalConfig();
};

const connect = () => {
  if (client) {
    return;
  }

  client = createWsClient({
    clientId,
    onOpen: () => {
      wsStatus.value = 'connected';
      client.send('GET_SETTINGS');
    },
    onMessage: (message) => {
      if (message.type === 'SETTINGS_DATA') {
        handleSettingsData(message.payload);
      } else if (message.type === 'SETTING_UPDATED') {
        handleSettingUpdated(message.payload);
      }
    },
    onClose: () => {
      wsStatus.value = 'disconnected';
    },
  });

  wsStatus.value = 'connecting';
  client.connect();
};

const refreshSettings = () => {
  ensureStarted();
  client?.send('GET_SETTINGS');
};

const start = () => {
  if (started) {
    return;
  }
  started = true;
  connect();
};

const stop = () => {
  started = false;
  client?.disconnect();
  client = null;
  wsStatus.value = 'disconnected';
};

const ensureStarted = () => {
  if (!started) {
    start();
  }
};

const saveSetting = (key, explicitValue = config.value[key]) => {
  ensureStarted();
  if (key === 'colors') {
    mergeConfig({ colors: explicitValue });
  } else if (key === 'custom_colors') {
    mergeConfig({
      custom_colors: explicitValue,
      colors: config.value.use_custom_theme ? explicitValue : config.value.colors,
    });
  } else if (key === 'theme') {
    mergeConfig({
      theme: explicitValue,
      use_custom_theme: false,
      colors: clonePalette(themeMap.value?.[explicitValue], config.value.colors),
    });
  } else if (key === 'use_custom_theme') {
    const useCustomTheme = Boolean(explicitValue);
    mergeConfig({
      use_custom_theme: useCustomTheme,
      colors: useCustomTheme
        ? clonePalette(config.value.custom_colors, config.value.colors)
        : clonePalette(themeMap.value?.[config.value.theme], config.value.colors),
    });
  } else if (key === 'ui_scale') {
    const nextUiScale = Math.min(125, Math.max(90, Number(explicitValue) || DEFAULT_CONFIG.ui_scale));
    mergeConfig({ ui_scale: nextUiScale });
    applyGlobalConfig();
    client?.send('UPDATE_SETTING', { key, value: nextUiScale });
    return;
  } else {
    mergeConfig({ [key]: explicitValue });
  }
  applyGlobalConfig();
  client?.send('UPDATE_SETTING', { key, value: explicitValue });
};

const saveCustomColors = (colors = config.value.custom_colors) => {
  const nextColors = clonePalette(colors, config.value.custom_colors);
  mergeConfig({
    custom_colors: nextColors,
    colors: config.value.use_custom_theme ? nextColors : config.value.colors,
  });
  applyGlobalConfig();
  ensureStarted();
  client?.send('UPDATE_SETTING', { key: 'colors', value: nextColors });
};

const setTheme = (themeName) => {
  saveSetting('theme', themeName);
};

const setCustomMode = () => {
  saveSetting('use_custom_theme', true);
};

const changeLanguage = (language) => {
  saveSetting('language', language);
};

const themes = computed(() =>
  Object.keys(themeMap.value).length > 0
    ? Object.keys(themeMap.value).sort()
    : ['Default', 'Chrome', 'Classic', 'Coral cave', 'Dice', 'Eclipse', 'Eight', 'Estoty', 'Galaxy', 'Green', 'Hobbel', 'Sunset']
);

const currentPalette = computed(() => {
  if (config.value.use_custom_theme) {
    return config.value.custom_colors
      ? config.value.custom_colors.slice(0, 16)
      : config.value.colors.slice(0, 16);
  }
  return themeMap.value[config.value.theme] || [];
});

export function useAppSettingsStore() {
  if (!loaded.value) {
    config.value.ui_scale = Number(config.value.ui_scale ?? DEFAULT_CONFIG.ui_scale) || DEFAULT_CONFIG.ui_scale;
    applyGlobalConfig();
  }

  return {
    wsStatus,
    loaded,
    config,
    categories,
    themeMap,
    targetTiles,
    themes,
    currentPalette,
    start,
    stop,
    refreshSettings,
    saveSetting,
    saveCustomColors,
    setTheme,
    setCustomMode,
    changeLanguage,
  };
}
