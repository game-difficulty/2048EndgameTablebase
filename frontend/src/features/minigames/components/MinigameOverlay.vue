<template>
  <Teleport to="body">
    <div
      v-if="overlay?.open"
      class="fixed inset-0 z-[120] flex items-center justify-center px-4"
    >
      <div class="relative w-full max-w-md">
        <div class="pointer-events-none absolute inset-[-44px] rounded-[2.25rem] bg-slate-950/16 blur-2xl"></div>
        <div class="relative rounded-3xl border border-border-main/92 bg-bg-card/[0.998] p-6 shadow-[0_2px_10px_rgba(15,23,42,0.12)] ring-1 ring-white/14">
          <div class="flex items-center justify-between gap-3">
            <h3 class="font-black text-text-main ui-text-xl tracking-tight">{{ overlay.title }}</h3>
            <span v-if="overlay.level" class="pill-badge pill-badge-accent uppercase">{{ overlay.level }}</span>
          </div>
          <p class="mt-3 whitespace-pre-wrap text-text-main ui-body font-semibold">{{ overlay.message }}</p>
          <div class="mt-5 flex gap-2">
            <button
              v-if="overlay.type === 'gameOver'"
              type="button"
              class="action-btn btn-prominent flex-1 text-white"
              @click="$emit('new-game')"
            >
              {{ $t('buttons.newGame') }}
            </button>
            <button
              type="button"
              :class="[overlay.type === 'gameOver' ? 'action-btn flex-1' : 'action-btn btn-prominent flex-1']"
              @click="$emit('close')"
            >
              {{ $t('minigames.overlay.continue') }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
defineProps({
  overlay: {
    type: Object,
    default: () => ({ open: false }),
  },
});

defineEmits(['close', 'new-game', 'back-menu']);
</script>
