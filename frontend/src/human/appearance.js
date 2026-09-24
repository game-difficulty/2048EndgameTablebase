import { ref, watch } from 'vue';
import themes from '../../../docs_and_configs/themes.json';
import { resolveTileColors } from '../utils/tileColors.js';
import { readSharedTilePalette, writeSharedTilePalette } from '../utils/sharedTilePalette.js';
import { createLocalStorageStore } from '../services/storage/localStorageStore.js';
import { refreshLanguage } from './i18n.js';

// Same preference envelope, theme catalog and resolved palette cookie as the main site.
const preferences = createLocalStorageStore({ key: 'user-preferences', version: 1, defaultValue: {} });
const validColor = value => typeof value === 'string' && /^#[\da-f]{6}$/i.test(value);
export const tileStyle = value => value ? {
  background: `var(--color-tile-${value}, #000000)`, color: `var(--color-text-${value}, #f9f6f2)`,
} : { background: 'var(--empty-tile, #716e69)', color: 'var(--muted)' };

export function useHumanAppearance() {
  const themeName = ref('Default'), hide32k = ref(false), darkMode = ref(true);
  function applyDarkMode(value) {
    darkMode.value = value;
    if (value) document.documentElement.setAttribute('data-theme', 'dark');
    else document.documentElement.removeAttribute('data-theme');
  }
  function setDarkMode(value) {
    preferences.update(current => ({ ...current, dark_mode: !!value }));
    applyDarkMode(!!value);
  }
  function refresh() {
    refreshLanguage();
    const stored = preferences.read();
    applyDarkMode(typeof stored.dark_mode === 'boolean' ? stored.dark_mode : true);
    hide32k.value = !!stored.dis_32k;
    const configured = stored.use_custom_theme ? stored.custom_colors : themes[stored.theme] || stored.colors;
    const raw = Array.isArray(configured) && configured.length ? configured : themes.Default;
    const fallback = resolveTileColors(Array.from({ length: 36 }, (_, i) => validColor(raw[i]) ? raw[i] : '#000000'));
    const shared = readSharedTilePalette();
    const palette = Array.from({ length: 36 }, (_, i) => {
      const item = shared?.[i];
      return validColor(item?.background) ? { background: item.background, color: validColor(item.color) ? item.color : fallback[i].color } : fallback[i];
    });
    themeName.value = Object.keys(themes).find(name => themes[name].every((color, i) => palette[i].background.toLowerCase() === color.toLowerCase())) || 'custom';
    palette.forEach((color, i) => {
      document.documentElement.style.setProperty(`--color-tile-${2 ** (i + 1)}`, color.background);
      document.documentElement.style.setProperty(`--color-text-${2 ** (i + 1)}`, color.color);
    });
  }
  function chooseTheme(name) {
    if (!themes[name]) return;
    const colors = Array.from({ length: 36 }, (_, i) => themes[name][i] || '#000000');
    preferences.update(current => ({ ...current, theme: name, use_custom_theme: false, colors }));
    writeSharedTilePalette(resolveTileColors(colors)); refresh();
  }
  watch(hide32k, value => { if (preferences.read().dis_32k !== value) preferences.update(current => ({ ...current, dis_32k: value })); });
  refresh();
  return { themeName, themeNames: Object.keys(themes).sort(), hide32k, darkMode, setDarkMode, chooseTheme, refresh };
}
