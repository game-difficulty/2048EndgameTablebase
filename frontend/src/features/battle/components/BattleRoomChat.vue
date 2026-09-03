<template>
  <section class="battle-room-chat" :class="{ 'battle-room-chat-compact': compact }">
    <header class="battle-chat-header">
      <div>
        <span class="ui-caption font-black uppercase text-text-secondary">{{ $t('battle.chat.kicker') }}</span>
        <h2>{{ $t('battle.chat.title') }}</h2>
      </div>
      <span class="battle-chat-limit">{{ $t('battle.chat.messageLimit', { count: 50 }) }}</span>
    </header>

    <div
      ref="messageList"
      class="battle-chat-messages"
      role="log"
      aria-live="polite"
      :aria-label="$t('battle.chat.messagesLabel')"
      @scroll="handleScroll"
    >
      <p v-if="messages.length === 0" class="battle-chat-empty">
        {{ $t('battle.chat.empty') }}
      </p>
      <ol v-else>
        <li v-for="message in messages" :key="String(message.message_id)" class="battle-chat-message">
          <div class="battle-chat-message-meta">
            <strong>{{ speakerName(message) }} <small v-if="isBattleGuest(message)" class="battle-chat-guest-marker">{{ $t('battle.guest.marker') }}</small></strong>
            <time :datetime="message.created_at">{{ formatTime(message.created_at) }}</time>
          </div>
          <p>{{ message.content }}</p>
        </li>
      </ol>
      <button
        v-if="hasUnread"
        type="button"
        class="battle-chat-new-message"
        @click="scrollToBottom"
      >
        {{ $t('battle.chat.newMessages') }}
      </button>
    </div>

    <p v-if="noticeText" id="battle-room-chat-notice" class="battle-chat-notice" role="status">
      {{ noticeText }}
    </p>

    <form class="battle-chat-compose" @submit.prevent="submitMessage">
      <label class="sr-only" for="battle-room-chat-input">{{ $t('battle.chat.inputLabel') }}</label>
      <input
        id="battle-room-chat-input"
        ref="inputElement"
        :value="draft"
        type="text"
        inputmode="text"
        autocomplete="off"
        enterkeyhint="send"
        :placeholder="$t('battle.chat.placeholder')"
        :disabled="!connected || !canSpeak || cooldownSeconds > 0"
        :aria-describedby="noticeText ? 'battle-room-chat-notice' : undefined"
        @input="handleInput"
        @compositionstart="isComposing = true"
        @compositionend="finishComposition"
        @keydown.enter="handleEnter"
      >
      <span class="battle-chat-character-count" :class="{ 'is-full': characterCount >= maxCharacters }">
        {{ characterCount }}/{{ maxCharacters }}
      </span>
      <button type="submit" :disabled="!canSend">
        {{ cooldownSeconds > 0
          ? $t('battle.chat.retryIn', { seconds: cooldownSeconds })
          : $t('battle.chat.send') }}
      </button>
    </form>
  </section>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import {
  BATTLE_CHAT_MAX_CODE_POINTS,
  chatCodePointLength,
  truncateChatContent,
  validateChatContent,
} from '../core/chatState.js';
import { isBattleGuest } from '../core/battleActor.js';

const props = defineProps({
  messages: { type: Array, default: () => [] },
  notice: { type: Object, default: null },
  cooldownSeconds: { type: Number, default: 0 },
  connected: { type: Boolean, default: false },
  canSpeak: { type: Boolean, default: true },
  disabledCode: { type: String, default: 'CHAT_ROLE_NOT_ALLOWED' },
  compact: { type: Boolean, default: false },
});
const emit = defineEmits(['send']);

const { locale, t, te } = useI18n();
const draft = ref('');
const isComposing = ref(false);
const messageList = ref(null);
const inputElement = ref(null);
const hasUnread = ref(false);
const maxCharacters = BATTLE_CHAT_MAX_CODE_POINTS;

const characterCount = computed(() => chatCodePointLength(draft.value));
const canSend = computed(() => (
  props.connected
  && props.canSpeak
  && props.cooldownSeconds <= 0
  && validateChatContent(draft.value).ok
));
const noticeText = computed(() => {
  const code = String(props.notice?.code || '');
  if (!code) {
    if (props.canSpeak) return '';
    const disabledKey = props.disabledCode === 'CHAT_GUEST_NOT_ALLOWED'
      ? 'battle.chat.guestForbidden'
      : 'battle.chat.errors.CHAT_ROLE_NOT_ALLOWED';
    return t(disabledKey);
  }
  if (code === 'CHAT_GUEST_NOT_ALLOWED') return t('battle.chat.guestForbidden');
  if (code === 'CHAT_RATE_LIMITED') {
    return t('battle.chat.errors.CHAT_RATE_LIMITED', {
      seconds: props.cooldownSeconds || props.notice?.retry_after_seconds || 1,
    });
  }
  const key = `battle.chat.errors.${code}`;
  return te(key) ? t(key) : t('battle.chat.errors.CHAT_REJECTED');
});

const speakerName = (message) => (
  String(message?.display_name || '').trim() || t('battle.chat.unknownUser')
);
const formatTime = (value) => {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '--:--';
  return new Intl.DateTimeFormat(locale.value, {
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
};
const isNearBottom = () => {
  const element = messageList.value;
  if (!element) return true;
  return element.scrollHeight - element.scrollTop - element.clientHeight < 24;
};
const scrollToBottom = async () => {
  await nextTick();
  const element = messageList.value;
  if (element) element.scrollTop = element.scrollHeight;
  hasUnread.value = false;
};
const handleScroll = () => {
  if (isNearBottom()) hasUnread.value = false;
};
const applyInputValue = (value) => {
  const limited = truncateChatContent(value, maxCharacters);
  draft.value = limited;
  if (inputElement.value && inputElement.value.value !== limited) {
    inputElement.value.value = limited;
  }
};
const handleInput = (event) => {
  if (isComposing.value) {
    draft.value = event.target.value;
    return;
  }
  applyInputValue(event.target.value);
};
const finishComposition = (event) => {
  isComposing.value = false;
  applyInputValue(event.target.value);
};
const submitMessage = () => {
  if (!canSend.value) return;
  const { content } = validateChatContent(draft.value);
  emit('send', content);
  draft.value = '';
  if (inputElement.value) inputElement.value.value = '';
};
const handleEnter = (event) => {
  if (isComposing.value || event.isComposing || event.keyCode === 229) return;
  event.preventDefault();
  submitMessage();
};

watch(
  () => props.messages,
  async (nextMessages, previousMessages = []) => {
    const shouldFollow = previousMessages.length === 0 || isNearBottom();
    const previousLastId = previousMessages.at(-1)?.message_id;
    const nextLastId = nextMessages.at(-1)?.message_id;
    await nextTick();
    if (shouldFollow) {
      await scrollToBottom();
    } else if (String(previousLastId ?? '') !== String(nextLastId ?? '')) {
      hasUnread.value = true;
    }
  },
);
</script>

<style scoped>
.battle-room-chat {
  width: 100%;
  margin-top: 14px;
  padding: 15px;
  border: 1px solid var(--border-main);
  border-radius: 8px;
  background: var(--bg-card);
  box-shadow: 0 14px 32px rgba(0, 0, 0, 0.06);
}
.battle-chat-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 10px;
}
.battle-chat-header h2 {
  margin: 2px 0 0;
  color: var(--text-main);
  font-size: 17px;
  font-weight: 900;
  letter-spacing: 0;
}
.battle-chat-limit {
  color: var(--text-secondary);
  font-size: 11px;
  font-weight: 800;
}
.battle-chat-messages {
  position: relative;
  height: 220px;
  overflow-y: auto;
  overscroll-behavior: contain;
  border: 1px solid var(--border-main);
  border-radius: 7px;
  background: color-mix(in srgb, var(--bg-main) 68%, var(--bg-card));
  scrollbar-gutter: stable;
  -webkit-overflow-scrolling: touch;
}
.battle-room-chat-compact .battle-chat-messages { height: 170px; }
.battle-chat-messages ol { margin: 0; padding: 4px 12px; list-style: none; }
.battle-chat-message { padding: 8px 0; border-bottom: 1px solid var(--border-main); }
.battle-chat-message:last-child { border-bottom: 0; }
.battle-chat-message-meta { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; }
.battle-chat-message-meta strong {
  min-width: 0;
  overflow: hidden;
  color: var(--text-main);
  font-size: 12px;
  font-weight: 900;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.battle-chat-guest-marker { margin-left: 3px; color: var(--accent); font-size: 8px; font-weight: 900; }
.battle-chat-message-meta time {
  flex: 0 0 auto;
  color: var(--text-secondary);
  font: 700 10px/1 var(--font-mono, monospace);
}
.battle-chat-message p {
  margin: 4px 0 0;
  overflow-wrap: anywhere;
  color: var(--text-main);
  font-size: 13px;
  line-height: 1.45;
  white-space: pre-wrap;
}
.battle-chat-empty {
  height: 100%;
  display: grid;
  place-items: center;
  margin: 0;
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 700;
}
.battle-chat-new-message {
  position: sticky;
  bottom: 8px;
  display: block;
  min-height: 30px;
  margin: 0 auto 8px;
  padding: 0 12px;
  border: 1px solid var(--accent);
  border-radius: 999px;
  background: var(--bg-card);
  color: var(--accent);
  font-size: 11px;
  font-weight: 900;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.09);
}
.battle-chat-notice {
  margin: 8px 2px 0;
  color: #c84848;
  font-size: 11px;
  font-weight: 800;
}
.battle-chat-compose {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto auto;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
}
.battle-chat-compose input {
  width: 100%;
  min-width: 0;
  height: 40px;
  padding: 0 12px;
  border: 1px solid var(--border-main);
  border-radius: 7px;
  outline: none;
  background: var(--bg-main);
  color: var(--text-main);
  font-size: 13px;
  font-weight: 700;
}
.battle-chat-compose input:focus { border-color: var(--accent); }
.battle-chat-compose input:disabled { cursor: not-allowed; opacity: 0.62; }
.battle-chat-character-count {
  min-width: 32px;
  color: var(--text-secondary);
  font: 700 10px/1 var(--font-mono, monospace);
  text-align: right;
}
.battle-chat-character-count.is-full { color: var(--accent); }
.battle-chat-compose button {
  min-width: 82px;
  height: 40px;
  padding: 0 13px;
  border: 1px solid var(--btn-bg);
  border-radius: 7px;
  background: var(--btn-bg);
  color: white;
  font-size: 12px;
  font-weight: 900;
}
.battle-chat-compose button:disabled { cursor: not-allowed; opacity: 0.5; }
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
</style>
