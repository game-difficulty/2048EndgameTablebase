<template>
  <div class="auth-page">
    <section ref="panelRef" class="auth-panel">
      <div class="auth-heading">
        <div class="auth-kicker">2048 Endgame Tablebase</div>
        <h1>{{ $t(`auth.title.${mode}`) }}</h1>
      </div>

      <div class="auth-tabs">
        <button :class="{ active: mode === 'login' }" @click="mode = 'login'">{{ $t('auth.tabs.login') }}</button>
        <button :class="{ active: mode === 'register' }" @click="mode = 'register'">{{ $t('auth.tabs.register') }}</button>
      </div>

      <div
        v-if="message"
        class="auth-message"
        :class="{ error: messageType === 'error' }"
        :role="messageType === 'error' ? 'alert' : 'status'"
        aria-live="polite"
      >
        {{ message }}
      </div>

      <form class="auth-form" @submit.prevent="submit">
        <label v-if="mode === 'register'">
          <span>{{ $t('auth.fields.displayName') }}</span>
          <input v-model="displayName" autocomplete="username" :placeholder="$t('auth.placeholders.displayName')" maxlength="80" required />
        </label>

        <div v-if="mode === 'register'" class="auth-email-row">
          <label class="auth-email-field" :class="{ invalid: emailDomainUnsupported }">
            <span>{{ $t('auth.fields.email') }}</span>
            <input v-model="email" type="email" autocomplete="email" :placeholder="$t('auth.placeholders.email')" required />
          </label>
          <button
            type="button"
            class="auth-secondary auth-send-code"
            :disabled="sendCodeDisabled"
            :title="sendCodeDisabledReason"
            @click="sendCode"
          >
            {{ registerCooldownRemaining > 0 ? cooldownLabel(registerCooldownRemaining) : (sendingCode ? $t('auth.actions.sendingCode') : $t('auth.actions.sendCode')) }}
          </button>
          <p class="auth-field-hint" :class="{ error: sendCodeHintIsError }">
            {{ emailDomainHint }}
          </p>
        </div>

        <label v-else>
          <span>{{ $t('auth.fields.email') }}</span>
          <input v-model="email" type="email" autocomplete="email" :placeholder="$t('auth.placeholders.email')" required />
        </label>

        <label v-if="mode === 'register'">
          <span>{{ $t('auth.fields.inviteCode') }}</span>
          <input v-model="inviteCode" autocomplete="off" :placeholder="$t('auth.placeholders.inviteCode')" />
        </label>

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
    </section>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue';
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
const SUPPORTED_EMAIL_DOMAINS = [
  'qq.com',
  'foxmail.com',
  '163.com',
  '126.com',
  'yeah.net',
  'gmail.com',
  'outlook.com',
  'hotmail.com',
  'icloud.com',
];

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
const panelRef = ref(null);
const nowMs = ref(Date.now());
let cooldownTimer = null;

const showMessage = (text, type = 'info') => {
  message.value = text;
  messageType.value = type;
  if (text) {
    nextTick(() => {
      panelRef.value?.scrollTo?.({ top: 0, behavior: type === 'error' ? 'smooth' : 'auto' });
    });
  }
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
const supportedEmailDomainsText = computed(() => SUPPORTED_EMAIL_DOMAINS.join(', '));
const registrationEmailDomain = computed(() => {
  const value = String(email.value || '').trim().toLowerCase();
  const at = value.lastIndexOf('@');
  if (at < 0 || at === value.length - 1) {
    return '';
  }
  return value.slice(at + 1);
});
const emailDomainUnsupported = computed(() => (
  mode.value === 'register'
  && registrationEmailDomain.value
  && !SUPPORTED_EMAIL_DOMAINS.includes(registrationEmailDomain.value)
));
const displayNameMissing = computed(() => mode.value === 'register' && !String(displayName.value || '').trim());
const emailMissing = computed(() => mode.value === 'register' && !email.value);
const emailDomainMissing = computed(() => mode.value === 'register' && email.value && !registrationEmailDomain.value);
const sendCodeHintIsError = computed(() => (
  displayNameMissing.value
  || emailMissing.value
  || emailDomainMissing.value
  || emailDomainUnsupported.value
));
const emailDomainHint = computed(() => {
  if (displayNameMissing.value) {
    return t('auth.hints.usernameRequiredForCode');
  }
  if (emailMissing.value || emailDomainMissing.value) {
    return t('auth.hints.emailRequiredForCode');
  }
  if (emailDomainUnsupported.value) {
    return t('auth.hints.unsupportedEmailDomain', { domains: supportedEmailDomainsText.value });
  }
  return t('auth.hints.supportedEmailDomains', { domains: supportedEmailDomainsText.value });
});
const sendCodeDisabledReason = computed(() => (sendCodeDisabled.value ? emailDomainHint.value : ''));
const sendCodeDisabled = computed(() => (
  sendingCode.value
  || displayNameMissing.value
  || emailMissing.value
  || emailDomainMissing.value
  || emailDomainUnsupported.value
  || registerCooldownRemaining.value > 0
));
const submitDisabled = computed(() => (
  submitting.value
  || (mode.value === 'forgot' && resetCooldownRemaining.value > 0)
  || (mode.value === 'register' && emailDomainUnsupported.value)
  || (mode.value === 'register' && !String(displayName.value || '').trim())
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
  width: 100%;
  background: transparent;
}

.auth-panel {
  width: 100%;
  max-height: min(42rem, calc(100vh - 3rem));
  overflow-y: auto;
  border: 1px solid var(--border-main);
  border-radius: 22px;
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--bg-card) 94%, white 6%), var(--bg-card));
  padding: 1.75rem;
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.28);
  scrollbar-width: thin;
}

.auth-heading h1 {
  margin: 0.25rem 0 0;
  color: var(--text-main);
  font-size: 1.75rem;
  line-height: 1.12;
  font-weight: 900;
}

.auth-kicker,
.auth-form span {
  color: var(--text-secondary);
  font-size: 0.72rem;
  font-weight: 900;
  letter-spacing: 0;
}

.auth-tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.35rem;
  margin: 1.35rem 0 1.2rem;
  border: 1px solid var(--border-main);
  border-radius: 14px;
  background: color-mix(in srgb, var(--bg-main) 80%, transparent);
  padding: 0.35rem;
}

.auth-tabs button,
.auth-secondary,
.auth-primary,
.auth-link {
  border: 1px solid var(--border-main);
  border-radius: 10px;
  padding: 0.75rem 0.95rem;
  background: var(--bg-main);
  color: var(--text-main);
  font-weight: 900;
  transition:
    transform 120ms ease,
    border-color 160ms ease,
    background-color 160ms ease,
    box-shadow 160ms ease,
    color 160ms ease;
}

.auth-tabs button.active,
.auth-primary {
  background: var(--btn-bg);
  border-color: color-mix(in srgb, var(--btn-bg) 78%, var(--border-main));
  color: white;
  box-shadow: 0 10px 26px rgba(15, 23, 42, 0.14);
}

.auth-tabs button:hover:not(.active),
.auth-secondary:hover:not(:disabled),
.auth-link:hover {
  border-color: color-mix(in srgb, var(--accent) 48%, var(--border-main));
  color: var(--text-main);
}

.auth-primary:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 14px 30px rgba(15, 23, 42, 0.18);
}

.auth-tabs button:disabled,
.auth-secondary:disabled,
.auth-primary:disabled,
.auth-link:disabled {
  cursor: not-allowed;
  opacity: 0.48;
  filter: saturate(0.55);
  box-shadow: none;
  transform: none;
}

.auth-secondary:disabled,
.auth-primary:disabled {
  border-color: color-mix(in srgb, var(--border-main) 72%, transparent);
  background: color-mix(in srgb, var(--bg-main) 72%, var(--border-main) 28%);
  color: color-mix(in srgb, var(--text-secondary) 72%, transparent);
}

.auth-link {
  background: transparent;
  color: var(--text-secondary);
}

.auth-form {
  display: grid;
  gap: 0.82rem;
}

.auth-form label,
.auth-email-field {
  display: grid;
  gap: 0.38rem;
  min-width: 0;
}

.auth-form input {
  box-sizing: border-box;
  width: 100%;
  min-width: 0;
  height: 2.75rem;
  min-height: 2.75rem;
  border: 1px solid var(--border-main);
  border-radius: 11px;
  background: color-mix(in srgb, var(--bg-main) 88%, white 12%);
  color: var(--text-main);
  padding: 0 0.9rem;
  font-weight: 800;
  outline: none;
  transition:
    border-color 160ms ease,
    box-shadow 160ms ease,
    background-color 160ms ease;
}

.auth-form input:focus {
  border-color: color-mix(in srgb, var(--accent) 60%, var(--border-main));
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent);
}

.auth-email-field.invalid input {
  border-color: color-mix(in srgb, #ef4444 68%, var(--border-main));
}

.auth-form input:-webkit-autofill,
.auth-form input:-webkit-autofill:hover,
.auth-form input:-webkit-autofill:focus {
  -webkit-text-fill-color: var(--text-main);
  box-shadow: 0 0 0 1000px color-mix(in srgb, var(--bg-main) 88%, white 12%) inset;
  transition: background-color 5000s ease-in-out 0s;
}

.auth-email-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: end;
  column-gap: 0.95rem;
  row-gap: 0.6rem;
}

.auth-send-code {
  box-sizing: border-box;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  align-self: end;
  height: 2.75rem;
  min-height: 0;
  padding: 0 1.05rem;
  white-space: nowrap;
}

.auth-field-hint {
  grid-column: 1 / -1;
  margin: -0.22rem 0 0;
  color: var(--text-secondary);
  font-size: 0.72rem;
  font-weight: 800;
  line-height: 1.45;
}

.auth-field-hint.error {
  color: #ef4444;
}

.auth-message {
  position: sticky;
  top: 0;
  z-index: 2;
  margin: 0 0 0.95rem;
  border: 1px solid color-mix(in srgb, var(--border-main) 82%, transparent);
  border-radius: 12px;
  background: color-mix(in srgb, var(--bg-card) 94%, var(--bg-main) 6%);
  padding: 0.75rem 0.85rem;
  color: var(--text-secondary);
  font-weight: 800;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.1);
}

.auth-message.error {
  border-color: color-mix(in srgb, #ef4444 42%, var(--border-main));
  background: color-mix(in srgb, #ef4444 10%, var(--bg-card));
  color: #ef4444;
}

@media (max-width: 520px) {
  .auth-panel {
    max-height: calc(100vh - 1.5rem);
    border-radius: 18px;
    padding: 1.2rem;
  }

  .auth-heading h1 {
    font-size: 1.5rem;
  }

  .auth-email-row {
    grid-template-columns: 1fr;
    gap: 0.6rem;
  }

  .auth-send-code {
    width: 100%;
  }
}
</style>
