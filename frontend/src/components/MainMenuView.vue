<template>
  <div class="page-root pt-6">
    <div class="main-menu-container w-full max-w-7xl mx-auto p-6 flex flex-col items-center">
    <!-- Hero Header -->
    <div class="text-center mb-7 animate-fade-in">
      <h1 class="text-7xl font-black text-text-main tracking-tighter mb-2 italic drop-shadow-md">2048</h1>
      <p class="ui-metric text-text-secondary font-black letter-spacing-wide uppercase tracking-widest opacity-80">
        {{ currentSite === 'tables' ? $t('menu.tablesSite') : '2048 EndgameTablebase' }}
      </p>
    </div>

    <div class="home-entry-grid">
      <component v-for="(entry, index) in homeEntries" :key="entry.id"
        :is="entry.href || entry.site ? 'a' : 'button'"
        :href="entry.site ? siteUrl(entry.site).href : entry.href"
        :target="entry.href || entry.site ? '_blank' : undefined"
        :rel="entry.href || entry.site ? 'noopener noreferrer' : undefined"
        :class="['menu-card', index < 3 ? 'menu-card-primary' : 'menu-card-secondary']"
        @click="entry.tab && $emit('selectTab', entry.tab)">
        <div :class="['menu-card-icon-shell', index < 3 ? 'menu-card-icon-primary' : 'menu-card-icon-secondary']">
          <component :is="homeIcons[entry.icon]" class="menu-card-icon-svg" :stroke-width="1.8" aria-hidden="true" />
        </div>
        <h3>{{ $t(entry.title) }}</h3>
        <p class="menu-card-copy ui-body text-text-secondary opacity-70">{{ $t(entry.description) }}</p>
      </component>
    </div>

    <div class="menu-bottom-row menu-bottom-row-with-tools">
      <div class="menu-utility-actions menu-utility-directory">
        <AuxiliaryEntry v-for="entry in HOME_AUXILIARY_ENTRIES" :key="entry.id" :entry="entry" @navigate="$emit('navigate', $event)" />
        <button
          v-if="showBrowserModeButton"
          type="button"
          class="menu-utility-btn"
          @click="openBrowserMode"
        >
          {{ $t('menu.openBrowserMode') }}
        </button>
      </div>
    </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted, ref } from 'vue';
import AuxiliaryEntry from './AuxiliaryEntry.vue';
import { HOME_AUXILIARY_ENTRIES } from '../app/auxiliaryEntries.js';
import { Gamepad2, Layers, Grid2X2, Cpu, Settings, CircleHelp, Target, ClipboardCheck, Swords, Clapperboard, ChartNoAxesCombined } from '@lucide/vue';
import { currentSite, siteUrl, MAIN_HOME_ENTRIES, TABLES_HOME_ENTRIES } from '../app/siteProfile.js';
const homeEntries = currentSite === 'tables' ? TABLES_HOME_ENTRIES : MAIN_HOME_ENTRIES;
const homeIcons = { Gamepad2, Layers, Grid2X2, Cpu, Settings, CircleHelp, Target, ClipboardCheck, Swords, Clapperboard, ChartNoAxesCombined };

defineProps(['active']);
defineEmits(['selectTab', 'navigate']);

const showBrowserModeButton = ref(false);

const syncBrowserModeButtonVisibility = () => {
  showBrowserModeButton.value = !!window.pywebview?.api?.open_external_url;
};

const openBrowserMode = async () => {
  const targetUrl = `${window.location.origin}${window.location.pathname}${window.location.search}${window.location.hash}`;
  try {
    if (window.pywebview?.api?.open_external_url) {
      const opened = await window.pywebview.api.open_external_url(targetUrl);
      if (opened) {
        return;
      }
    }
  } catch (error) {
    console.error(error);
  }
  window.open(targetUrl, '_blank', 'noopener,noreferrer');
};

onMounted(() => {
  syncBrowserModeButtonVisibility();
  window.addEventListener('pywebviewready', syncBrowserModeButtonVisibility);
});

onUnmounted(() => {
  window.removeEventListener('pywebviewready', syncBrowserModeButtonVisibility);
});
</script>

<style scoped>
.home-entry-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); grid-auto-rows:1fr; gap:20px; width:100%; padding:0 16px; }
.home-entry-grid .menu-card { min-height:200px; border-radius:8px; padding:22px; display:flex; flex-direction:column; align-items:center; text-align:center; text-decoration:none; transition:transform .2s,box-shadow .2s; border-style:solid; border-width:1px 1px 6px; }
.home-entry-grid .menu-card:hover { transform:translateY(-4px); }
.home-entry-grid .menu-card-icon-shell { margin-bottom:14px; }
.home-entry-grid h3 { margin:0 0 8px; font-size:23px; font-weight:900; color:var(--text-main); letter-spacing:0; }

.menu-card {
  user-select: none;
  box-shadow: 0 18px 34px rgba(0, 0, 0, 0.08);
}

.menu-bottom-row {
  position: relative;
  width: 100%;
  margin-top: 2.5rem;
  padding-inline: 1rem;
}

.menu-utility-actions {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 0.75rem;
  margin: 1rem auto 0;
}

.menu-utility-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid var(--border-main);
  border-radius: 999px;
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--bg-card) 92%, white 8%) 0%, var(--bg-card) 100%);
  color: var(--text-main);
  padding: 0.85rem 1.4rem;
  font-size: var(--font-ui-sm);
  font-weight: 900;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
  transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, color 0.2s ease;
  text-decoration: none;
  gap: 0.55rem;
}

.menu-utility-icon {
  width: 1.15rem;
  height: 1.15rem;
  flex: 0 0 auto;
}

.menu-utility-btn:hover {
  transform: translateY(-2px);
  border-color: var(--accent);
  color: var(--accent);
  box-shadow: 0 14px 28px rgba(0, 0, 0, 0.12);
}

.menu-card-primary {
  background:
    var(--menu-primary-overlay),
    var(--menu-primary-accent),
    var(--bg-card);
  border-color: var(--border-main);
  border-bottom-width: 8px;
  border-bottom-color: var(--menu-primary-border-bottom);
}

.menu-card-primary:hover {
  box-shadow: 0 18px 36px var(--menu-primary-hover-shadow);
}

.menu-card-secondary {
  background:
    var(--menu-secondary-overlay),
    var(--menu-secondary-accent),
    var(--bg-card);
  border-color: var(--border-main);
  border-bottom-width: 8px;
  border-bottom-color: var(--menu-secondary-border-bottom);
}

.menu-card-secondary:hover {
  box-shadow: 0 18px 34px var(--menu-secondary-hover-shadow);
}

.menu-card-minigames {
  background:
    var(--menu-minigames-overlay),
    var(--menu-minigames-accent),
    var(--bg-card);
  border-color: color-mix(in srgb, var(--border-main) 88%, white 12%);
  border-bottom-width: 8px;
  border-bottom-color: var(--menu-minigames-border-bottom);
  box-shadow: 0 16px 30px color-mix(in srgb, var(--menu-minigames-hover-shadow) 72%, transparent);
}

.menu-card-minigames:hover {
  box-shadow: 0 18px 34px var(--menu-minigames-hover-shadow);
}

.menu-card-minigames-icon {
  background: linear-gradient(135deg, var(--menu-minigames-icon-start) 0%, var(--menu-minigames-icon-end) 100%);
  color: var(--menu-minigames-icon-text);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.28),
    0 5px 15px color-mix(in srgb, var(--menu-minigames-hover-shadow) 90%, transparent);
}

.menu-card-icon-shell {
  width: 3.5rem;
  height: 3.5rem;
  border-radius: 0.85rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.menu-card-icon-primary {
  background: linear-gradient(135deg, var(--menu-primary-icon-start) 0%, var(--menu-primary-icon-end) 100%);
  color: var(--menu-primary-icon-text);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.22),
    0 7px 16px color-mix(in srgb, var(--menu-primary-hover-shadow) 92%, transparent);
}

.menu-card-icon-secondary {
  background: linear-gradient(135deg, var(--menu-secondary-icon-start) 0%, var(--menu-secondary-icon-end) 100%);
  color: var(--menu-secondary-icon-text);
  box-shadow: 0 5px 15px color-mix(in srgb, var(--menu-secondary-hover-shadow) 88%, transparent);
}

.menu-card-icon-svg {
  width: 2.1rem;
  height: 2.1rem;
  display: block;
}

.menu-card-copy {
  min-height: 2.9rem;
  line-height: 1.35;
}

.letter-spacing-wide {
  letter-spacing: 0.15em;
}

.menu-bottom-row {
  min-height: 3rem;
  display:flex;
  flex-direction:column;
  margin-top:1rem;
}

.menu-utility-directory { order:-1; }

.menu-bottom-row-with-tools {
  padding-inline: 1rem;
}

.menu-utility-actions {
  position: static;
  flex-direction: row;
  align-items: center;
  margin: 8px auto 0;
  transform: none;
}

.menu-utility-actions-left { left: 1rem; }
.menu-utility-actions-right { right: 1rem; }
</style>
