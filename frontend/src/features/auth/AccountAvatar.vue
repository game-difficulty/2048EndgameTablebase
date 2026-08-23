<template>
  <component
    :is="editable ? 'button' : 'span'"
    :type="editable ? 'button' : undefined"
    :class="['account-avatar-shell', size === 'large' ? 'large' : '', supporter ? 'supporter' : '', editable ? 'editable' : '']"
    :title="editable ? $t('profile.avatar.change') : undefined"
    :aria-label="editable ? $t('profile.avatar.change') : undefined"
    @click="editable && $emit('edit')"
  >
    <span class="account-avatar-media">
      <img v-if="avatarUrl && !imageFailed" :src="avatarUrl" alt="" @error="imageFailed = true" />
      <span v-else>{{ initials }}</span>
    </span>
    <span v-if="supporter" class="account-supporter-mark" aria-hidden="true" />
    <span v-if="editable" class="account-avatar-edit" aria-hidden="true">✎</span>
  </component>
</template>

<script setup>
import { computed, ref, watch } from 'vue';

const props = defineProps({
  user: { type: Object, default: null },
  supporter: { type: Boolean, default: false },
  editable: { type: Boolean, default: false },
  size: { type: String, default: 'small' },
});

defineEmits(['edit']);

const imageFailed = ref(false);
const avatarUrl = computed(() => String(props.user?.profile?.avatar_url || ''));
const initials = computed(() => {
  const value = String(props.user?.display_name || props.user?.email || '?').trim();
  const parts = value.split(/\s+/).filter(Boolean);
  if (parts.length >= 2) {
    return `${parts[0][0] || ''}${parts[1][0] || ''}`.toUpperCase();
  }
  return value.slice(0, 2).toUpperCase();
});

watch(avatarUrl, () => {
  imageFailed.value = false;
});
</script>

<style scoped>
.account-avatar-shell {
  display: inline-flex;
  position: relative;
  width: 1.5rem;
  height: 1.5rem;
  align-items: center;
  justify-content: center;
  flex: 0 0 auto;
  border: 1px solid color-mix(in srgb, var(--accent) 26%, transparent);
  border-radius: 999px;
  background: color-mix(in srgb, var(--accent) 18%, var(--bg-card));
  color: var(--accent);
  font-size: 0.72rem;
  font-weight: 950;
  line-height: 1;
  padding: 0;
}

.account-avatar-shell.large {
  width: 3.25rem;
  height: 3.25rem;
  font-size: 0.9rem;
}

.account-avatar-media {
  display: flex;
  width: 100%;
  height: 100%;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  border-radius: inherit;
}

.account-avatar-media img {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: cover;
}

.account-avatar-shell.supporter {
  border-color: color-mix(in srgb, var(--accent) 70%, var(--border-main));
  box-shadow:
    0 0 0 2px color-mix(in srgb, var(--accent) 14%, transparent),
    inset 0 0 0 1px color-mix(in srgb, var(--success) 18%, transparent);
}

.account-supporter-mark {
  position: absolute;
  right: -0.16rem;
  bottom: -0.14rem;
  z-index: 3;
  width: 0.7rem;
  height: 0.7rem;
  border: 2px solid color-mix(in srgb, var(--bg-card) 94%, white);
  border-radius: 999px;
  background: linear-gradient(135deg, var(--accent), color-mix(in srgb, var(--success) 72%, var(--accent)));
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--accent) 38%, transparent);
}

.account-supporter-mark::after {
  content: "";
  position: absolute;
  inset: 20%;
  background: var(--bg-card);
  clip-path: polygon(50% 0, 62% 34%, 100% 50%, 62% 66%, 50% 100%, 38% 66%, 0 50%, 38% 34%);
}

.large .account-supporter-mark {
  right: -0.12rem;
  bottom: -0.12rem;
  width: 0.88rem;
  height: 0.88rem;
}

.editable {
  cursor: pointer;
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
}

.editable:hover,
.editable:focus-visible {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent);
  outline: none;
}

.editable:active {
  transform: scale(0.97);
}

.account-avatar-edit {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: inherit;
  background: color-mix(in srgb, var(--bg-main) 68%, transparent);
  color: var(--text-main);
  font-size: 1rem;
  opacity: 0;
  transition: opacity 160ms ease;
}

.editable:hover .account-avatar-edit,
.editable:focus-visible .account-avatar-edit {
  opacity: 1;
}
</style>

