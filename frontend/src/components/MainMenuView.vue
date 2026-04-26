<template>
  <div class="page-root pt-6">
    <div class="main-menu-container w-full max-w-5xl mx-auto p-6 flex flex-col items-center">
    <!-- Hero Header -->
    <div class="text-center mb-7 animate-fade-in">
      <h1 class="text-7xl font-black text-text-main tracking-tighter mb-2 italic drop-shadow-md">2048</h1>
      <p class="ui-metric text-text-secondary font-black letter-spacing-wide uppercase tracking-widest opacity-80">
        2048 EndgameTablebase
      </p>
    </div>

    <!-- Menu Grid -->
    <div class="grid w-full grid-cols-1 gap-6 px-4 md:grid-cols-3 md:auto-rows-fr">
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

    <!-- Footer / Support -->
    <div class="mt-10 text-center shadow-sm px-5 py-4 rounded-[1.75rem] bg-bg-card backdrop-blur-md border border-border-main">
      <p class="text-text-secondary font-black ui-control uppercase tracking-widest flex items-center gap-2 justify-center">
        {{ $t('menu.githubNote') }}
        <a href="https://github.com/game-difficulty/2048EndgameTablebase" target="_blank"
          rel="noopener noreferrer"
          class="text-prominent font-black hover:underline flex items-center gap-1 group">
          <svg class="w-5 h-5 fill-current transition-transform group-hover:scale-110" viewBox="0 0 24 24">
            <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/>
          </svg>
          GitHub
        </a>
      </p>
      <p class="mt-1 text-text-secondary ui-caption font-bold">
        {{ $t('menu.sponsorNotePrefix') }}<a href="https://2048next.cn" target="_blank" rel="noopener noreferrer"
          class="text-prominent font-black hover:underline">2048next.cn</a>{{ $t('menu.sponsorNoteSuffix') }}
      </p>
    </div>
    </div>
  </div>
</template>

<script setup>
defineProps(['active']);
defineEmits(['selectTab']);
</script>

<style scoped>
.menu-card {
  user-select: none;
  box-shadow: 0 18px 34px rgba(0, 0, 0, 0.08);
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
</style>
