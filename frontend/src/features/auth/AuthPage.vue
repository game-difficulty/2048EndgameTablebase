<template>
  <div class="auth-page">
    <section class="auth-panel">
      <div class="auth-heading">
        <div class="auth-kicker">2048 Endgame Tablebase</div>
        <h1>{{ $t(`auth.title.${mode}`) }}</h1>
      </div>

      <div class="auth-tabs">
        <button :class="{ active: mode === 'login' }" @click="mode = 'login'">{{ $t('auth.tabs.login') }}</button>
        <button :class="{ active: mode === 'register' }" @click="mode = 'register'">{{ $t('auth.tabs.register') }}</button>
      </div>

      <form class="auth-form" @submit.prevent="submit">
        <label v-if="mode === 'register'">
          <span>{{ $t('auth.fields.inviteCode') }}</span>
          <input v-model="inviteCode" autocomplete="off" :placeholder="$t('auth.placeholders.inviteCode')" required />
        </label>

        <label>
          <span>{{ $t('auth.fields.email') }}</span>
          <input v-model="email" type="email" autocomplete="email" :placeholder="$t('auth.placeholders.email')" required />
        </label>

        <div v-if="mode === 'register'" class="auth-send-row">
          <button type="button" class="auth-secondary" :disabled="sendCodeDisabled" @click="sendCode">
            {{ registerCooldownRemaining > 0 ? cooldownLabel(registerCooldownRemaining) : (sendingCode ? $t('auth.actions.sendingCode') : $t('auth.actions.sendCode')) }}
          </button>
        </div>

        <label v-if="mode === 'register' || mode === 'reset'">
          <span>{{ $t('auth.fields.emailCode') }}</span>
          <input v-model="verificationCode" inputmode="numeric" autocomplete="one-time-code" :placeholder="$t('auth.placeholders.emailCode')" required />
        </label>

        <label v-if="mode !== 'forgot'">
          <span>{{ mode === 'reset' ? $t('auth.fields.newPassword') : $t('auth.fields.password') }}</span>
          <input
            v-model="password"
            type="password"
            :autocomplete="mode === 'login' ? 'current-password' : 'new-password'"
            :placeholder="$t('auth.placeholders.password')"
            required
          />
        </label>

        <label v-if="mode === 'register'">
          <span>{{ $t('auth.fields.displayName') }}</span>
          <input v-model="displayName" autocomplete="name" :placeholder="$t('auth.placeholders.displayName')" />
        </label>

        <button type="submit" class="auth-primary" :disabled="submitDisabled">
          {{ submitting ? $t('auth.actions.pleaseWait') : submitLabel }}
        </button>

        <button
          v-if="mode === 'login'"
          type="button"
          class="auth-link"
          @click="mode = 'forgot'"
        >
          {{ $t('auth.actions.forgotPassword') }}
        </button>

        <button
          v-if="mode === 'forgot' || mode === 'reset'"
          type="button"
          class="auth-link"
          @click="mode = 'login'"
        >
          {{ $t('auth.actions.backToLogin') }}
        </button>
      </form>

      <div v-if="message" class="auth-message" :class="{ error: messageType === 'error' }">
        {{ message }}
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { authClient } from '../../services/auth/authClient';

const emit = defineEmits(['authenticated']);
const props = defineProps({
  initialMode: {
    type: String,
    default: 'login',
  },
});
const { t } = useI18n();
const COOLDOWN_SECONDS = 5 * 60;
const COOLDOWN_STORAGE_KEY = '2048tables:auth-code-cooldowns:v1';

const normalizeMode = (value) => (
  ['login', 'register', 'forgot', 'reset'].includes(value) ? value : 'login'
);

const mode = ref(normalizeMode(props.initialMode));
const email = ref('');
const password = ref('');
const inviteCode = ref('');
const verificationCode = ref('');
const displayName = ref('');
const submitting = ref(false);
const sendingCode = ref(false);
const message = ref('');
const messageType = ref('info');
const nowMs = ref(Date.now());
let cooldownTimer = null;

const showMessage = (text, type = 'info') => {
  message.value = text;
  messageType.value = type;
};

const submitLabel = computed(() => {
  if (mode.value === 'register') return t('auth.actions.register');
  if (mode.value === 'forgot') {
    return resetCooldownRemaining.value > 0
      ? cooldownLabel(resetCooldownRemaining.value)
      : t('auth.actions.sendResetCode');
  }
  if (mode.value === 'reset') return t('auth.actions.resetPassword');
  return t('auth.actions.login');
});

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

const registerCooldownRemaining = computed(() => cooldownRemaining('register'));
const resetCooldownRemaining = computed(() => cooldownRemaining('password_reset'));
const sendCodeDisabled = computed(() => (
  sendingCode.value || !email.value || !inviteCode.value || registerCooldownRemaining.value > 0
));
const submitDisabled = computed(() => (
  submitting.value || (mode.value === 'forgot' && resetCooldownRemaining.value > 0)
));

const applyServerCooldown = (error, purpose) => {
  if (error?.detail?.code !== 'EMAIL_CODE_COOLDOWN') {
    return false;
  }
  const retryAfter = Number(error.detail.retry_after_seconds || COOLDOWN_SECONDS);
  setCooldown(purpose, retryAfter);
  showMessage(cooldownLabel(retryAfter), 'error');
  return true;
};

watch(
  () => props.initialMode,
  (nextMode) => {
    mode.value = normalizeMode(nextMode);
    showMessage('');
  }
);

watch(mode, () => {
  showMessage('');
});

const sendCode = async () => {
  sendingCode.value = true;
  showMessage('');
  try {
    const result = await authClient.sendEmailCode({
      email: email.value,
      invite_code: inviteCode.value,
    });
    setCooldown('register');
    showMessage(result.dev_code ? t('auth.messages.devCode', { code: result.dev_code }) : t('auth.messages.codeSent'));
  } catch (error) {
    if (!applyServerCooldown(error, 'register')) {
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
    if (mode.value === 'forgot') {
      await authClient.requestPasswordReset({ email: email.value });
      setCooldown('password_reset');
      mode.value = 'reset';
      showMessage(t('auth.messages.resetCodeSent'));
      return;
    }
    const result = mode.value === 'login'
      ? await authClient.login({ email: email.value, password: password.value })
      : mode.value === 'reset'
        ? await authClient.resetPassword({
          email: email.value,
          verification_code: verificationCode.value,
          new_password: password.value,
        })
        : await authClient.register({
        email: email.value,
        password: password.value,
        invite_code: inviteCode.value,
        verification_code: verificationCode.value,
        display_name: displayName.value,
      });
    emit('authenticated', result.user);
  } catch (error) {
    if (!applyServerCooldown(error, mode.value === 'forgot' ? 'password_reset' : 'register')) {
      showMessage(error.message || String(error), 'error');
    }
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
.auth-page {
  min-height: 100%;
  display: grid;
  place-items: center;
  padding: 2rem;
  background: var(--bg-main);
}

.auth-panel {
  width: min(28rem, 100%);
  border: 1px solid var(--border-main);
  border-radius: 24px;
  background: var(--bg-card);
  padding: 2rem;
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.22);
}

.auth-heading h1 {
  margin: 0.35rem 0 0;
  color: var(--text-main);
  font-size: 2rem;
  font-weight: 900;
}

.auth-kicker,
.auth-form span {
  color: var(--text-secondary);
  font-size: 0.72rem;
  font-weight: 900;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.auth-tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  margin: 1.5rem 0;
}

.auth-tabs button,
.auth-secondary,
.auth-primary,
.auth-link {
  border: 1px solid var(--border-main);
  border-radius: 12px;
  padding: 0.85rem 1rem;
  background: var(--bg-main);
  color: var(--text-main);
  font-weight: 900;
}

.auth-tabs button.active,
.auth-primary {
  background: var(--btn-bg);
  color: white;
}

.auth-link {
  background: transparent;
  color: var(--text-secondary);
}

.auth-link:hover {
  color: var(--accent);
}

.auth-form {
  display: grid;
  gap: 1rem;
}

.auth-form label {
  display: grid;
  gap: 0.45rem;
}

.auth-form input {
  min-height: 2.85rem;
  border: 1px solid var(--border-main);
  border-radius: 12px;
  background: var(--bg-main);
  color: var(--text-main);
  padding: 0 0.9rem;
  font-weight: 800;
  outline: none;
}

.auth-send-row {
  display: flex;
  justify-content: flex-end;
}

.auth-message {
  margin-top: 1rem;
  color: var(--text-secondary);
  font-weight: 800;
}

.auth-message.error {
  color: #ef4444;
}
</style>
