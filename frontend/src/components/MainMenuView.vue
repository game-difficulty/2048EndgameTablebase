<template>
  <div class="page-root pt-6">
    <div class="main-menu-container w-full max-w-7xl mx-auto p-6 flex flex-col items-center">
    <!-- Hero Header -->
    <div class="text-center mb-7 animate-fade-in">
      <h1 class="text-7xl font-black text-text-main tracking-tighter mb-2 italic drop-shadow-md">2048</h1>
      <p class="ui-metric text-text-secondary font-black letter-spacing-wide uppercase tracking-widest opacity-80">
        2048 EndgameTablebase
      </p>
    </div>

    <!-- Menu Grid -->
    <div class="grid w-full grid-cols-4 auto-rows-fr gap-5 px-4">
      <!-- Play Game Card -->
      <button @click="$emit('selectTab', 'GamerView')"
        class="menu-card menu-card-primary h-full min-h-[220px] rounded-2xl p-6 transition-all duration-300 transform hover:-translate-y-2 flex flex-col items-center text-center group border backdrop-blur-sm">
        <div class="menu-card-icon-shell menu-card-icon-primary mb-4 group-hover:scale-110 transition-transform duration-500">
          <svg class="menu-card-icon-svg" viewBox="0 0 64 64" aria-hidden="true">
            <rect x="12" y="24" width="40" height="18" rx="9" fill="currentColor" opacity="0.16" />
            <path d="M20 26c-3.2 0-5.8 2.4-6.1 5.6l-1 8.6c-.2 2.1 1.4 3.8 3.5 3.8h5.7c1.5 0 2.8-.8 3.5-2.1l2.3-4.2h8.2l2.3 4.2c.7 1.3 2 2.1 3.5 2.1h5.7c2.1 0 3.7-1.8 3.5-3.8l-1-8.6C50 28.4 47.4 26 44.2 26H20Z" fill="currentColor"/>
            <rect x="21" y="31" width="2.8" height="9" rx="1.4" fill="white"/>
            <rect x="17.9" y="34.1" width="9" height="2.8" rx="1.4" fill="white"/>
            <circle cx="40.5" cy="33.8" r="2.2" fill="white"/>
            <circle cx="45.2" cy="37.8" r="2.2" fill="white"/>
          </svg>
        </div>
        <h3 class="text-[1.45rem] font-black text-text-main mb-2 tracking-tight">{{ $t('menu.play') }}</h3>
        <p class="menu-card-copy ui-body text-text-secondary opacity-70">{{ $t('menu.descriptions.play') }}</p>
      </button>

      <!-- Trainer Card -->
      <button @click="$emit('selectTab', 'TrainerView')"
        class="menu-card menu-card-primary h-full min-h-[220px] rounded-2xl p-6 transition-all duration-300 transform hover:-translate-y-2 flex flex-col items-center text-center group border backdrop-blur-sm">
        <div class="menu-card-icon-shell menu-card-icon-primary mb-4 group-hover:scale-110 transition-transform duration-500">
          <svg class="menu-card-icon-svg" viewBox="0 0 64 64" aria-hidden="true">
            <circle cx="32" cy="32" r="17.5" fill="currentColor" opacity="0.16"/>
            <circle cx="32" cy="32" r="13" stroke="currentColor" stroke-width="4" fill="none"/>
            <circle cx="32" cy="32" r="4.7" fill="currentColor"/>
            <path d="M32 11v7M32 46v7M11 32h7M46 32h7" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>
          </svg>
        </div>
        <h3 class="text-[1.45rem] font-black text-text-main mb-2 tracking-tight">{{ $t('menu.practice') }}</h3>
        <p class="menu-card-copy ui-body text-text-secondary opacity-70">{{ $t('menu.descriptions.practice') }}</p>
      </button>

      <!-- Tester Card -->
      <button @click="$emit('selectTab', 'TesterView')"
        class="menu-card menu-card-primary h-full min-h-[220px] rounded-2xl p-6 transition-all duration-300 transform hover:-translate-y-2 flex flex-col items-center text-center group border backdrop-blur-sm">
        <div class="menu-card-icon-shell menu-card-icon-primary mb-4 group-hover:scale-110 transition-transform duration-500">
          <svg class="menu-card-icon-svg" viewBox="0 0 64 64" aria-hidden="true">
            <rect x="17" y="13" width="30" height="38" rx="6" fill="currentColor" opacity="0.16"/>
            <rect x="20" y="16" width="24" height="32" rx="5" stroke="currentColor" stroke-width="4" fill="none"/>
            <rect x="26" y="11" width="12" height="7" rx="3.5" fill="currentColor"/>
            <path d="M26 29l3.8 3.8L38 24.6" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M26 39h12" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>
          </svg>
        </div>
        <h3 class="text-[1.45rem] font-black text-text-main mb-2 tracking-tight">{{ $t('menu.test') }}</h3>
        <p class="menu-card-copy ui-body text-text-secondary opacity-70">{{ $t('menu.descriptions.test') }}</p>
      </button>

      <!-- Battle -->
      <button @click="$emit('selectTab', 'BattleView')"
        class="menu-card menu-card-primary h-full min-h-[220px] rounded-2xl p-6 transition-all duration-300 transform hover:-translate-y-2 flex flex-col items-center text-center group border backdrop-blur-sm">
        <div class="menu-card-icon-shell menu-card-icon-primary mb-4 group-hover:scale-110 transition-transform duration-500">
          <svg class="menu-card-icon-svg" viewBox="0 0 64 64" aria-hidden="true">
            <path d="M13 19h16v26H13zM35 19h16v26H35z" fill="currentColor" opacity="0.18"/>
            <path d="M18 26h6M18 32h6M18 38h6M40 26h6M40 32h6M40 38h6" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/>
            <path d="M28 32h8" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>
            <circle cx="32" cy="32" r="3.2" fill="currentColor"/>
          </svg>
        </div>
        <h3 class="text-[1.45rem] font-black text-text-main mb-2 tracking-tight">{{ $t('menu.battle') }}</h3>
        <p class="menu-card-copy ui-body text-text-secondary opacity-70">{{ $t('menu.descriptions.battle') }}</p>
      </button>

      <!-- Minigames -->
      <button @click="$emit('selectTab', 'MinigamesView')"
        class="menu-card menu-card-minigames h-full min-h-[220px] rounded-2xl p-6 transition-all duration-300 transform hover:-translate-y-2 flex flex-col items-center text-center group border backdrop-blur-sm">
        <div class="menu-card-icon-shell menu-card-minigames-icon mb-4 group-hover:scale-110 transition-transform duration-500">
          <svg class="menu-card-icon-svg" viewBox="0 0 64 64" aria-hidden="true">
            <rect x="15" y="15" width="13" height="13" rx="4" fill="currentColor"/>
            <rect x="36" y="15" width="13" height="13" rx="4" fill="currentColor" opacity="0.84"/>
            <rect x="15" y="36" width="13" height="13" rx="4" fill="currentColor" opacity="0.84"/>
            <rect x="36" y="36" width="13" height="13" rx="4" fill="currentColor"/>
            <path d="M31.5 19.5 35 16l3.5 3.5L35 23z" fill="white" opacity="0.95"/>
          </svg>
        </div>
        <h3 class="text-[1.45rem] font-black text-text-main mb-2 tracking-tight">{{ $t('menu.minigames') }}</h3>
        <p class="menu-card-copy ui-body text-text-secondary opacity-70">{{ $t('menu.descriptions.minigames') }}</p>
      </button>

      <!-- Leaderboards -->
      <button @click="$emit('selectTab', 'LeaderboardsView')"
        class="menu-card menu-card-secondary h-full min-h-[220px] rounded-2xl p-6 transition-all duration-300 transform hover:-translate-y-2 flex flex-col items-center text-center group border backdrop-blur-sm">
        <div class="menu-card-icon-shell menu-card-icon-secondary mb-4 group-hover:scale-110 transition-transform duration-500">
          <svg class="menu-card-icon-svg" viewBox="0 0 64 64" aria-hidden="true">
            <path d="M22 13h20v10c0 8-3.8 13.5-10 16.2C25.8 36.5 22 31 22 23V13Z" fill="currentColor" opacity="0.2"/>
            <path d="M22 18h-7v4c0 6.2 3.8 10.2 10 11M42 18h7v4c0 6.2-3.8 10.2-10 11M32 39v8M22 52h20M27 47h10" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
          </svg>
        </div>
        <h3 class="text-[1.45rem] font-black text-text-main mb-2 tracking-tight">{{ $t('menu.leaderboards') }}</h3>
        <p class="menu-card-copy ui-body text-text-secondary opacity-70">{{ $t('menu.descriptions.leaderboards') }}</p>
      </button>

      <!-- Settings -->
      <button @click="$emit('selectTab', 'SettingsView')"
        class="menu-card menu-card-secondary h-full min-h-[220px] rounded-2xl p-6 transition-all duration-300 transform hover:-translate-y-2 flex flex-col items-center text-center group border backdrop-blur-sm">
        <div class="menu-card-icon-shell menu-card-icon-secondary mb-4 group-hover:scale-110 transition-transform duration-500">
          <svg class="menu-card-icon-svg" viewBox="0 0 64 64" aria-hidden="true">
            <path d="M27.6 12h8.8l1.2 5.3a16 16 0 0 1 3.9 1.6l4.7-2.7 6.2 6.2-2.7 4.7a16 16 0 0 1 1.6 3.9l5.3 1.2v8.8l-5.3 1.2a16 16 0 0 1-1.6 3.9l2.7 4.7-6.2 6.2-4.7-2.7a16 16 0 0 1-3.9 1.6L36.4 52h-8.8l-1.2-5.3a16 16 0 0 1-3.9-1.6l-4.7 2.7-6.2-6.2 2.7-4.7a16 16 0 0 1-1.6-3.9L7.4 40v-8.8l5.3-1.2a16 16 0 0 1 1.6-3.9l-2.7-4.7 6.2-6.2 4.7 2.7a16 16 0 0 1 3.9-1.6L27.6 12Z" fill="currentColor" opacity="0.22"/>
            <circle cx="32" cy="36" r="8.2" stroke="currentColor" stroke-width="4" fill="none"/>
          </svg>
        </div>
        <h3 class="text-[1.45rem] font-black text-text-main mb-2 tracking-tight">{{ $t('menu.settings') }}</h3>
        <p class="menu-card-copy ui-body text-text-secondary opacity-70">{{ $t('menu.descriptions.settings') }}</p>
      </button>

      <!-- Help -->
      <button @click="$emit('selectTab', 'HelpView')"
        class="menu-card menu-card-secondary h-full min-h-[220px] rounded-2xl p-6 transition-all duration-300 transform hover:-translate-y-2 flex flex-col items-center text-center group border backdrop-blur-sm">
        <div class="menu-card-icon-shell menu-card-icon-secondary mb-4 group-hover:scale-110 transition-transform duration-500">
          <svg class="menu-card-icon-svg" viewBox="0 0 64 64" aria-hidden="true">
            <circle cx="32" cy="32" r="19" fill="currentColor" opacity="0.18"/>
            <path d="M25.5 25.5c0-4 3-6.8 7.3-6.8 4.1 0 6.9 2.5 6.9 6 0 2.7-1.2 4.4-4 6.2-2.7 1.8-3.2 2.7-3.2 4.8v1" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
            <circle cx="32" cy="44.8" r="2.4" fill="currentColor"/>
          </svg>
        </div>
        <h3 class="text-[1.45rem] font-black text-text-main mb-2 tracking-tight">{{ $t('menu.help') }}</h3>
        <p class="menu-card-copy ui-body text-text-secondary opacity-70">{{ $t('menu.descriptions.help') }}</p>
      </button>
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
