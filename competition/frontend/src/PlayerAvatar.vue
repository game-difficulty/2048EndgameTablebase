<script setup>
import { computed, ref, watch } from 'vue';

const props = defineProps({ person: { type: Object, default: null } });
const failed = ref(false);
const mainSiteUrl = String(import.meta.env.VITE_MAIN_SITE_URL || 'https://2048tables.online/');
const avatarUrl = computed(() => {
  const path = String(props.person?.avatar_url || '');
  if (!path) return '';
  return new URL(path, mainSiteUrl).toString();
});
const initials = computed(() => {
  const name = String(props.person?.display_name || '?').trim();
  const parts = name.split(/\s+/).filter(Boolean);
  return (parts.length > 1 ? `${parts[0][0]}${parts[1][0]}` : name.slice(0, 2)).toUpperCase();
});
watch(avatarUrl, () => { failed.value = false; });
</script>

<template>
  <span class="player-avatar" aria-hidden="true">
    <img v-if="avatarUrl && !failed" :src="avatarUrl" alt="" @error="failed = true" />
    <span v-else>{{ initials }}</span>
  </span>
</template>

<style scoped>
.player-avatar { display: grid; flex: none; place-items: center; width: var(--player-avatar-size, 36px); height: var(--player-avatar-size, 36px); overflow: hidden; color: #7e6850; background: #e9e1d6; border: 1px solid rgba(126, 104, 80, .22); border-radius: 50%; font-size: calc(var(--player-avatar-size, 36px) * .3); font-weight: 700; line-height: 1; }
.player-avatar img { display: block; width: 100%; height: 100%; object-fit: cover; }
.player-avatar > span { overflow: hidden; max-width: 100%; white-space: nowrap; }
</style>
