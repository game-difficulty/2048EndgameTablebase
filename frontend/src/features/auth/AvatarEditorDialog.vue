<template>
  <div
    v-if="open"
    class="fixed inset-0 z-[124] flex items-center justify-center bg-slate-950/42 p-6 backdrop-blur-sm"
    @click.self="$emit('close')"
  >
    <section class="avatar-dialog">
      <header class="avatar-dialog-header">
        <div>
          <div class="profile-kicker">{{ $t('profile.title') }}</div>
          <h2>{{ $t('profile.avatar.title') }}</h2>
        </div>
        <button type="button" class="action-btn-small close-button" @click="$emit('close')">
          {{ $t('common.close') }}
        </button>
      </header>

      <div v-if="!uploadEnabled" class="cooldown-panel">
        <strong>{{ $t('profile.avatar.disabledTitle') }}</strong>
        <span>{{ $t('profile.avatar.disabled') }}</span>
      </div>

      <div v-else-if="!canChange" class="cooldown-panel">
        <strong>{{ $t('profile.cooldown.title') }}</strong>
        <span>{{ $t('profile.cooldown.avatar', { date: availableDate }) }}</span>
      </div>

      <div v-else class="avatar-editor-body">
        <input
          ref="fileInput"
          class="sr-only"
          type="file"
          accept="image/*"
          @change="handleFile"
        />

        <div v-if="imageUrl" ref="cropViewport" class="crop-viewport" @pointerdown="startDrag">
          <img
            ref="sourceImage"
            :src="imageUrl"
            alt=""
            draggable="false"
            :style="imageStyle"
            @load="handleImageLoaded"
          />
          <span class="crop-guide" aria-hidden="true" />
        </div>
        <div v-else class="avatar-current">
          <AccountAvatar :user="user" :supporter="supporter" size="large" />
          <span>{{ $t('profile.avatar.chooseHint') }}</span>
        </div>

        <label v-if="imageUrl" class="zoom-row">
          <span>{{ $t('profile.avatar.zoom') }}</span>
          <input v-model.number="zoom" type="range" min="1" max="3" step="0.01" @input="clampPan" />
        </label>

        <div v-if="message" :class="['profile-message', messageType === 'error' ? 'error' : '']">
          {{ message }}
        </div>

        <div class="avatar-actions">
          <button type="button" class="action-btn-small" :disabled="submitting" @click="fileInput?.click()">
            {{ imageUrl ? $t('profile.avatar.chooseAnother') : $t('profile.avatar.choose') }}
          </button>
          <button
            v-if="user?.profile?.avatar_url && !imageUrl"
            type="button"
            class="action-btn-small danger-button"
            :disabled="submitting"
            @click="removeAvatar"
          >
            {{ $t('profile.avatar.remove') }}
          </button>
          <button
            v-if="imageUrl"
            type="button"
            class="action-btn-small surface-prominent text-white"
            :disabled="submitting || !imageReady"
            @click="saveAvatar"
          >
            {{ submitting ? $t('auth.actions.pleaseWait') : $t('profile.avatar.save') }}
          </button>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { profileClient } from '../../services/auth/profileClient';
import AccountAvatar from './AccountAvatar.vue';

const props = defineProps({
  open: { type: Boolean, default: false },
  user: { type: Object, default: null },
  supporter: { type: Boolean, default: false },
});
const emit = defineEmits(['close', 'saved']);
const { t, locale } = useI18n();

const VIEWPORT_SIZE = 280;
const OUTPUT_SIZE = 512;
const MAX_SOURCE_BYTES = 16 * 1024 * 1024;
const fileInput = ref(null);
const cropViewport = ref(null);
const sourceImage = ref(null);
const imageUrl = ref('');
const imageReady = ref(false);
const naturalWidth = ref(0);
const naturalHeight = ref(0);
const zoom = ref(1);
const panX = ref(0);
const panY = ref(0);
const submitting = ref(false);
const message = ref('');
const messageType = ref('info');
let dragState = null;

const uploadEnabled = computed(() => props.user?.profile?.can_upload_avatar !== false);
const canChange = computed(() => props.user?.profile?.can_change_avatar !== false);
const availableDate = computed(() => {
  const raw = props.user?.profile?.avatar_change_available_at;
  if (!raw) return '';
  const date = new Date(raw);
  if (!Number.isFinite(date.getTime())) return raw;
  return new Intl.DateTimeFormat(locale.value === 'zh' ? 'zh-CN' : 'en-US', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date);
});

const baseScale = computed(() => {
  if (!naturalWidth.value || !naturalHeight.value) return 1;
  return Math.max(VIEWPORT_SIZE / naturalWidth.value, VIEWPORT_SIZE / naturalHeight.value);
});
const renderedWidth = computed(() => naturalWidth.value * baseScale.value * zoom.value);
const renderedHeight = computed(() => naturalHeight.value * baseScale.value * zoom.value);
const imageStyle = computed(() => ({
  width: `${renderedWidth.value}px`,
  height: `${renderedHeight.value}px`,
  transform: `translate(calc(-50% + ${panX.value}px), calc(-50% + ${panY.value}px))`,
}));

const revokeImageUrl = () => {
  if (imageUrl.value) URL.revokeObjectURL(imageUrl.value);
  imageUrl.value = '';
};

const resetEditor = () => {
  revokeImageUrl();
  imageReady.value = false;
  naturalWidth.value = 0;
  naturalHeight.value = 0;
  zoom.value = 1;
  panX.value = 0;
  panY.value = 0;
  message.value = '';
  messageType.value = 'info';
  if (fileInput.value) fileInput.value.value = '';
};

watch(() => props.open, (open) => {
  if (!open) resetEditor();
});

const clampPan = () => {
  const maxX = Math.max(0, (renderedWidth.value - VIEWPORT_SIZE) / 2);
  const maxY = Math.max(0, (renderedHeight.value - VIEWPORT_SIZE) / 2);
  panX.value = Math.min(maxX, Math.max(-maxX, panX.value));
  panY.value = Math.min(maxY, Math.max(-maxY, panY.value));
};

const handleFile = (event) => {
  const file = event?.target?.files?.[0];
  if (!file) return;
  message.value = '';
  if (!String(file.type || '').startsWith('image/')) {
    message.value = t('profile.errors.invalidAvatar');
    messageType.value = 'error';
    return;
  }
  if (file.size > MAX_SOURCE_BYTES) {
    message.value = t('profile.errors.sourceTooLarge');
    messageType.value = 'error';
    return;
  }
  revokeImageUrl();
  imageReady.value = false;
  zoom.value = 1;
  panX.value = 0;
  panY.value = 0;
  imageUrl.value = URL.createObjectURL(file);
};

const handleImageLoaded = () => {
  naturalWidth.value = sourceImage.value?.naturalWidth || 0;
  naturalHeight.value = sourceImage.value?.naturalHeight || 0;
  imageReady.value = naturalWidth.value >= 32 && naturalHeight.value >= 32;
  if (!imageReady.value) {
    message.value = t('profile.errors.invalidAvatar');
    messageType.value = 'error';
  }
  clampPan();
};

const startDrag = (event) => {
  if (!imageReady.value || !cropViewport.value) return;
  event.preventDefault();
  const rect = cropViewport.value.getBoundingClientRect();
  dragState = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    panX: panX.value,
    panY: panY.value,
    scaleX: VIEWPORT_SIZE / Math.max(1, rect.width),
    scaleY: VIEWPORT_SIZE / Math.max(1, rect.height),
  };
  cropViewport.value.setPointerCapture?.(event.pointerId);
  window.addEventListener('pointermove', continueDrag);
  window.addEventListener('pointerup', stopDrag, { once: true });
};

const continueDrag = (event) => {
  if (!dragState || event.pointerId !== dragState.pointerId) return;
  panX.value = dragState.panX + (event.clientX - dragState.startX) * dragState.scaleX;
  panY.value = dragState.panY + (event.clientY - dragState.startY) * dragState.scaleY;
  clampPan();
};

const stopDrag = () => {
  dragState = null;
  window.removeEventListener('pointermove', continueDrag);
};

const canvasBlob = async () => {
  const canvas = document.createElement('canvas');
  canvas.width = OUTPUT_SIZE;
  canvas.height = OUTPUT_SIZE;
  const context = canvas.getContext('2d', { alpha: false });
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
  const factor = OUTPUT_SIZE / VIEWPORT_SIZE;
  const left = (VIEWPORT_SIZE - renderedWidth.value) / 2 + panX.value;
  const top = (VIEWPORT_SIZE - renderedHeight.value) / 2 + panY.value;
  context.drawImage(
    sourceImage.value,
    left * factor,
    top * factor,
    renderedWidth.value * factor,
    renderedHeight.value * factor
  );
  const makeBlob = (type, quality) => new Promise((resolve) => canvas.toBlob(resolve, type, quality));
  let blob = await makeBlob('image/webp', 0.84);
  if (!blob || blob.type !== 'image/webp' || blob.size > 512 * 1024) {
    blob = await makeBlob('image/jpeg', 0.86);
  }
  if (!blob || blob.size > 512 * 1024) throw new Error(t('profile.errors.encodeFailed'));
  return blob;
};

const translatedError = (error) => {
  const code = error?.detail?.code;
  if (code === 'PROFILE_CHANGE_COOLDOWN') return t('profile.errors.cooldown');
  if (code === 'AVATAR_UPLOAD_DISABLED') return t('profile.avatar.disabled');
  if (code === 'INVALID_AVATAR') return t('profile.errors.invalidAvatar');
  if (code === 'PROFILE_RATE_LIMIT') return t('profile.errors.rateLimit');
  return error?.message || String(error);
};

const saveAvatar = async () => {
  submitting.value = true;
  message.value = '';
  try {
    const blob = await canvasBlob();
    const result = await profileClient.updateAvatar(blob);
    emit('saved', result.user);
  } catch (error) {
    message.value = translatedError(error);
    messageType.value = 'error';
  } finally {
    submitting.value = false;
  }
};

const removeAvatar = async () => {
  if (!window.confirm(t('profile.avatar.removeConfirm'))) return;
  submitting.value = true;
  message.value = '';
  try {
    const result = await profileClient.removeAvatar();
    emit('saved', result.user);
  } catch (error) {
    message.value = translatedError(error);
    messageType.value = 'error';
  } finally {
    submitting.value = false;
  }
};

onUnmounted(() => {
  stopDrag();
  revokeImageUrl();
});
</script>

<style scoped>
.avatar-dialog {
  width: min(35rem, calc(100vw - 3rem));
  max-height: calc(100vh - 3rem);
  overflow-y: auto;
  border: 1px solid var(--border-main);
  border-radius: 20px;
  background: var(--bg-card);
  color: var(--text-main);
  padding: 1.5rem;
  box-shadow: 0 24px 80px rgba(15, 23, 42, 0.32);
}

.avatar-dialog-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: start;
  gap: 1rem;
}

.avatar-dialog h2 {
  margin: 0.35rem 0 0;
  font-size: 1.55rem;
  font-weight: 900;
  line-height: 1.15;
}

.profile-kicker {
  color: var(--text-secondary);
  font-size: 0.72rem;
  font-weight: 900;
  text-transform: uppercase;
}

.close-button {
  width: auto;
  min-width: 4.5rem;
}

.avatar-editor-body {
  display: grid;
  justify-items: center;
  gap: 1rem;
  margin-top: 1.25rem;
}

.crop-viewport {
  position: relative;
  width: 280px;
  height: 280px;
  max-width: 100%;
  overflow: hidden;
  border: 1px solid var(--border-main);
  border-radius: 12px;
  background: var(--bg-main);
  cursor: grab;
  touch-action: none;
  user-select: none;
}

.crop-viewport:active { cursor: grabbing; }

.crop-viewport img {
  position: absolute;
  left: 50%;
  top: 50%;
  max-width: none;
  pointer-events: none;
}

.crop-guide {
  position: absolute;
  inset: 8px;
  border: 2px solid rgba(255, 255, 255, 0.82);
  border-radius: 50%;
  box-shadow: 0 0 0 100px rgba(15, 23, 42, 0.32);
  pointer-events: none;
}

.avatar-current {
  display: grid;
  justify-items: center;
  gap: 0.75rem;
  padding: 1rem;
  color: var(--text-secondary);
  font-size: 0.78rem;
  font-weight: 800;
}

.zoom-row {
  display: grid;
  width: min(280px, 100%);
  grid-template-columns: auto minmax(0, 1fr);
  align-items: center;
  gap: 0.75rem;
  color: var(--text-secondary);
  font-size: 0.75rem;
  font-weight: 800;
}

.zoom-row input { width: 100%; accent-color: var(--accent); }

.avatar-actions {
  display: flex;
  width: 100%;
  justify-content: center;
  flex-wrap: wrap;
  gap: 0.65rem;
}

.danger-button { color: #ef4444; }

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

.profile-message {
  width: 100%;
  margin-top: 0;
}

.profile-message.error {
  border-color: color-mix(in srgb, #ef4444 55%, var(--border-main));
  color: #ef4444;
}
</style>
