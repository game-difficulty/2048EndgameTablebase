<template>
  <div
    v-if="open"
    class="fixed inset-0 z-[122] flex items-center justify-center bg-slate-950/42 p-6 backdrop-blur-sm"
    @click.self="$emit('close')"
  >
    <section class="security-dialog">
      <div class="flex items-start justify-between gap-4">
        <div>
          <div class="auth-kicker">{{ $t('auth.account.title') }}</div>
          <h2>{{ title }}</h2>
        </div>
        <button type="button" class="action-btn-small" @click="$emit('close')">
          {{ $t('common.close') }}
        </button>
      </div>

      <form class="security-form" @submit.prevent="submit">
        <template v-if="mode === 'changePassword'">
          <label>
            <span>{{ $t('auth.fields.currentPassword') }}</span>
            <input
              v-model="currentPassword"
              type="password"
              autocomplete="current-password"
              :placeholder="$t('auth.placeholders.currentPassword')"
              required
            />
          </label>
          <label>
            <span>{{ $t('auth.fields.newPassword') }}</span>
            <input
              v-model="newPassword"
              type="password"
              autocomplete="new-password"
              :placeholder="$t('auth.placeholders.password')"
              required
            />
          </label>
        </template>

        <template v-else>
          <p class="security-warning">{{ $t('auth.deactivate.warning') }}</p>
          <div class="security-send-row">
            <button
              type="button"
              class="auth-secondary"
              :disabled="sendDeactivateCodeDisabled"
              @click="sendDeactivateCode"
            >
              {{ deactivateCooldownRemaining > 0 ? cooldownLabel(deactivateCooldownRemaining) : (sendingCode ? $t('auth.actions.sendingCode') : $t('auth.actions.sendDeactivateCode')) }}
            </button>
          </div>
          <label>
            <span>{{ $t('auth.fields.emailCode') }}</span>
            <input
              v-model="verificationCode"
              inputmode="numeric"
              autocomplete="one-time-code"
              :placeholder="$t('auth.placeholders.emailCode')"
              required
            />
          </label>
          <label>
            <span>{{ $t('auth.fields.password') }}</span>
            <input
              v-model="password"
              type="password"
              autocomplete="current-password"
              :placeholder="$t('auth.placeholders.password')"
              required
            />
          </label>
          <label>
            <span>{{ $t('auth.fields.confirmText') }}</span>
            <input
              v-model="confirmText"
              autocomplete="off"
              :placeholder="$t('auth.deactivate.confirmPlaceholder')"
              required
            />
          </label>
        </template>

        <div v-if="message" class="security-message" :class="{ error: messageType === 'error' }">
          {{ message }}
        </div>

        <button
          type="submit"
          :class="mode === 'deactivate' ? 'auth-danger' : 'auth-primary'"
          :disabled="submitting"
        >
          {{ submitting ? $t('auth.actions.pleaseWait') : submitLabel }}
        </button>
      </form>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { authClient } from '../../services/auth/authClient';

const props = defineProps({
  open: {
    type: Boolean,
    default: false,
  },
  mode: {
    type: String,
    default: 'changePassword',
  },
});

const emit = defineEmits(['close', 'success', 'deactivated']);
const { t } = useI18n();
const COOLDOWN_SECONDS = 5 * 60;
const COOLDOWN_STORAGE_KEY = '2048tables:auth-code-cooldowns:v1';
const DEACTIVATE_PURPOSE = 'account_deactivate';

const currentPassword = ref('');
const newPassword = ref('');
const password = ref('');
const confirmText = ref('');
const verificationCode = ref('');
const submitting = ref(false);
const sendingCode = ref(false);
const message = ref('');
const messageType = ref('info');
const nowMs = ref(Date.now());
let cooldownTimer = null;

const title = computed(() => (
  props.mode === 'deactivate'
    ? t('auth.account.deactivateAccount')
    : t('auth.account.changePassword')
));

const submitLabel = computed(() => (
  props.mode === 'deactivate'
    ? t('auth.actions.deactivate')
    : t('auth.actions.changePassword')
));

const resetForm = () => {
  currentPassword.value = '';
  newPassword.value = '';
  password.value = '';
  confirmText.value = '';
  verificationCode.value = '';
  message.value = '';
  messageType.value = 'info';
};

watch(
  () => [props.open, props.mode],
  () => {
    resetForm();
  }
);

const showMessage = (text, type = 'info') => {
  message.value = text;
  messageType.value = type;
};

const readCooldowns = () => {
  if (typeof window === 'undefined') return {};
  try {
    return JSON.parse(window.localStorage.getItem(COOLDOWN_STORAGE_KEY) || '{}') || {};
  } catch {
    return {};
  }
};

const writeCooldowns = (cooldowns) => {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(COOLDOWN_STORAGE_KEY, JSON.stringify(cooldowns || {}));
};

const setCooldown = (purpose, seconds = COOLDOWN_SECONDS) => {
  const cooldowns = readCooldowns();
  cooldowns[purpose] = Date.now() + Math.max(1, Number(seconds || COOLDOWN_SECONDS)) * 1000;
  writeCooldowns(cooldowns);
  nowMs.value = Date.now();
};

const cooldownRemaining = (purpose) => {
  const until = Number(readCooldowns()[purpose] || 0);
  return Math.max(0, Math.ceil((until - nowMs.value) / 1000));
};

const cooldownLabel = (seconds) => {
  const remaining = Math.max(0, Number(seconds || 0));
  const minutes = Math.floor(remaining / 60);
  const rest = String(remaining % 60).padStart(2, '0');
  return t('auth.actions.codeCooldown', { time: `${minutes}:${rest}` });
};

const deactivateCooldownRemaining = computed(() => cooldownRemaining(DEACTIVATE_PURPOSE));
const sendDeactivateCodeDisabled = computed(() => (
  sendingCode.value || deactivateCooldownRemaining.value > 0
));

const applyServerCooldown = (error) => {
  if (error?.detail?.code !== 'EMAIL_CODE_COOLDOWN') {
    return false;
  }
  const retryAfter = Number(error.detail.retry_after_seconds || COOLDOWN_SECONDS);
  setCooldown(DEACTIVATE_PURPOSE, retryAfter);
  showMessage(cooldownLabel(retryAfter), 'error');
  return true;
};

const sendDeactivateCode = async () => {
  sendingCode.value = true;
  showMessage('');
  try {
    const result = await authClient.requestDeactivationCode();
    setCooldown(DEACTIVATE_PURPOSE);
    showMessage(result.dev_code ? t('auth.messages.devCode', { code: result.dev_code }) : t('auth.messages.deactivateCodeSent'));
  } catch (error) {
    if (!applyServerCooldown(error)) {
      showMessage(error.message || String(error), 'error');
    }
  } finally {
    sendingCode.value = false;
  }
};

const submit = async () => {
  submitting.value = true;
  showMessage('');
  try {
    if (props.mode === 'deactivate') {
      await authClient.deactivate({
        password: password.value,
        confirm: confirmText.value,
        verification_code: verificationCode.value,
      });
      emit('deactivated');
      return;
    }
    const result = await authClient.changePassword({
      current_password: currentPassword.value,
      new_password: newPassword.value,
    });
    showMessage(t('auth.messages.passwordChanged'));
    emit('success', result.user);
  } catch (error) {
    showMessage(error.message || String(error), 'error');
  } finally {
    submitting.value = false;
  }
};

onMounted(() => {
  cooldownTimer = window.setInterval(() => {
    nowMs.value = Date.now();
  }, 1000);
});

onUnmounted(() => {
  if (cooldownTimer) {
    window.clearInterval(cooldownTimer);
    cooldownTimer = null;
  }
});
</script>

<style scoped>
.security-dialog {
  width: min(28rem, 100%);
  border: 1px solid var(--border-main);
  border-radius: 24px;
  background: var(--bg-card);
  color: var(--text-main);
  padding: 1.5rem;
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.32);
}

.security-dialog h2 {
  margin: 0.35rem 0 0;
  font-size: 1.45rem;
  font-weight: 900;
}

.auth-kicker,
.security-form span {
  color: var(--text-secondary);
  font-size: 0.72rem;
  font-weight: 900;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.security-form {
  display: grid;
  gap: 1rem;
  margin-top: 1.35rem;
}

.security-send-row {
  display: flex;
  justify-content: flex-end;
}

.security-form label {
  display: grid;
  gap: 0.45rem;
}

.security-form input {
  min-height: 2.85rem;
  border: 1px solid var(--border-main);
  border-radius: 12px;
  background: var(--bg-main);
  color: var(--text-main);
  padding: 0 0.9rem;
  font-weight: 800;
  outline: none;
}

.auth-primary,
.auth-danger,
.auth-secondary {
  min-height: 2.85rem;
  border: 1px solid var(--border-main);
  border-radius: 12px;
  padding: 0.85rem 1rem;
  color: white;
  font-weight: 900;
}

.auth-primary {
  background: var(--btn-bg);
}

.auth-danger {
  border-color: rgba(239, 68, 68, 0.5);
  background: #dc2626;
}

.auth-secondary {
  background: var(--bg-main);
  color: var(--text-main);
}

.security-warning,
.security-message {
  margin: 0;
  color: var(--text-secondary);
  font-weight: 800;
}

.security-message.error {
  color: #ef4444;
}
</style>
