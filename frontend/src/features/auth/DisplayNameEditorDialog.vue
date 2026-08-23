<template>
  <div
    v-if="open"
    class="fixed inset-0 z-[124] flex items-center justify-center bg-slate-950/42 p-6 backdrop-blur-sm"
    @click.self="$emit('close')"
  >
    <section class="profile-dialog">
      <header class="profile-dialog-header">
        <div>
          <div class="profile-kicker">{{ $t('profile.title') }}</div>
          <h2>{{ $t('profile.displayName.title') }}</h2>
        </div>
        <button type="button" class="action-btn-small close-button" @click="$emit('close')">
          {{ $t('common.close') }}
        </button>
      </header>

      <div v-if="!canChange" class="cooldown-panel">
        <strong>{{ $t('profile.cooldown.title') }}</strong>
        <span>{{ $t('profile.cooldown.displayName', { date: availableDate }) }}</span>
      </div>

      <form v-else class="profile-form" @submit.prevent="submit">
        <label>
          <span>{{ $t('profile.displayName.label') }}</span>
          <input
            v-model="displayName"
            autocomplete="nickname"
            maxlength="24"
            :placeholder="$t('profile.displayName.placeholder')"
            required
          />
        </label>
        <div class="field-meta">
          <span>{{ $t('profile.displayName.rules') }}</span>
          <span>{{ displayName.length }}/24</span>
        </div>
        <div v-if="message" :class="['profile-message', messageType === 'error' ? 'error' : '']">
          {{ message }}
        </div>
        <button type="submit" class="auth-primary" :disabled="submitting || !displayName.trim()">
          {{ submitting ? $t('auth.actions.pleaseWait') : $t('profile.displayName.save') }}
        </button>
      </form>
    </section>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { profileClient } from '../../services/auth/profileClient';

const props = defineProps({
  open: { type: Boolean, default: false },
  user: { type: Object, default: null },
});
const emit = defineEmits(['close', 'saved']);
const { t, locale } = useI18n();

const displayName = ref('');
const submitting = ref(false);
const message = ref('');
const messageType = ref('info');

const canChange = computed(() => props.user?.profile?.can_change_display_name !== false);
const availableDate = computed(() => {
  const raw = props.user?.profile?.display_name_change_available_at;
  if (!raw) return '';
  const date = new Date(raw);
  if (!Number.isFinite(date.getTime())) return raw;
  return new Intl.DateTimeFormat(locale.value === 'zh' ? 'zh-CN' : 'en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date);
});

watch(
  () => [props.open, props.user?.display_name],
  () => {
    displayName.value = String(props.user?.display_name || '');
    message.value = '';
    messageType.value = 'info';
  },
  { immediate: true }
);

const translatedError = (error) => {
  const code = error?.detail?.code;
  if (code === 'DISPLAY_NAME_TAKEN') return t('profile.errors.nameTaken');
  if (code === 'PROFILE_CHANGE_COOLDOWN') return t('profile.errors.cooldown');
  if (code === 'PROFILE_RATE_LIMIT') return t('profile.errors.rateLimit');
  if (code === 'INVALID_PROFILE_UPDATE') return t('profile.errors.invalidName');
  return error?.message || String(error);
};

const submit = async () => {
  submitting.value = true;
  message.value = '';
  try {
    const result = await profileClient.updateDisplayName(displayName.value);
    emit('saved', result.user);
  } catch (error) {
    message.value = translatedError(error);
    messageType.value = 'error';
  } finally {
    submitting.value = false;
  }
};
</script>

<style scoped>
.profile-dialog {
  width: min(32rem, calc(100vw - 3rem));
  border: 1px solid var(--border-main);
  border-radius: 20px;
  background: var(--bg-card);
  color: var(--text-main);
  padding: 1.5rem;
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.32);
}

.profile-dialog-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: start;
  gap: 1rem;
}

.profile-dialog h2 {
  margin: 0.35rem 0 0;
  font-size: 1.55rem;
  font-weight: 900;
  line-height: 1.15;
}

.profile-kicker,
.profile-form label > span {
  color: var(--text-secondary);
  font-size: 0.72rem;
  font-weight: 900;
  text-transform: uppercase;
}

.close-button {
  width: auto;
  min-width: 4.5rem;
}

.profile-form {
  display: grid;
  gap: 0.9rem;
  margin-top: 1.35rem;
}

.profile-form label {
  display: grid;
  gap: 0.45rem;
}

.profile-form input {
  width: 100%;
  border: 1px solid var(--border-main);
  border-radius: 12px;
  background: var(--bg-main);
  color: var(--text-main);
  padding: 0.82rem 0.95rem;
  font-weight: 800;
  outline: none;
}

.profile-form input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 14%, transparent);
}

.field-meta {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  color: var(--text-secondary);
  font-size: 0.72rem;
  font-weight: 700;
}

.cooldown-panel,
.profile-message {
  display: grid;
  gap: 0.35rem;
  margin-top: 1.25rem;
  border: 1px solid var(--border-main);
  border-radius: 12px;
  background: color-mix(in srgb, var(--accent) 8%, var(--bg-main));
  padding: 1rem;
  font-size: 0.82rem;
}

.profile-message.error {
  border-color: color-mix(in srgb, #ef4444 55%, var(--border-main));
  color: #ef4444;
}
</style>
