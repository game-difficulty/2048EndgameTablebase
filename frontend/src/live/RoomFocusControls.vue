<template>
  <button
    type="button"
    class="room-focus-enter"
    :disabled="disabled"
    :aria-pressed="active"
    :title="disabled ? t('请先关闭小窗','Close the mini player first') : t('仅显示直播内容','Show only the stream content')"
    @click="emit('update:active', true)"
  >
    <Maximize2 :size="16" aria-hidden="true" />
    <span>{{ t('全屏观看','Focus view') }}</span>
  </button>
  <Teleport to="body">
    <button
      v-if="active"
      type="button"
      class="room-focus-exit"
      :aria-label="t('退出全屏观看','Exit focus view')"
      :title="t('退出全屏观看（Esc）','Exit focus view (Esc)')"
      @click="emit('update:active', false)"
    >
      <Minimize2 :size="20" aria-hidden="true" />
    </button>
  </Teleport>
</template>

<script setup>
import { onBeforeUnmount, onMounted } from 'vue';
import { Maximize2, Minimize2 } from '@lucide/vue';

const props = defineProps({
  active: Boolean,
  disabled: Boolean,
  lang: { type: String, default: 'zh' },
});
const emit = defineEmits(['update:active']);
const t = (zh, en) => props.lang === 'zh' ? zh : en;

function onKeydown(event) {
  if (props.active && event.key === 'Escape') emit('update:active', false);
}

onMounted(() => document.addEventListener('keydown', onKeydown));
onBeforeUnmount(() => document.removeEventListener('keydown', onKeydown));
</script>

<style scoped>
.room-focus-enter {
  display:inline-flex;
  flex-flow:row nowrap;
  align-items:center;
  width:max-content;
  white-space:nowrap;
}
.room-focus-enter > svg { flex:0 0 auto; }
.room-focus-enter > span { flex:0 0 auto;white-space:nowrap; }
.room-focus-exit {
  position:fixed;
  z-index:1000;
  top:12px;
  left:50%;
  top:max(12px,env(safe-area-inset-top));
  transform:translateX(-50%);
  display:grid;
  place-items:center;
  width:42px;
  height:42px;
  padding:0;
  border:1px solid color-mix(in srgb,var(--border-main) 80%,transparent);
  border-radius:8px;
  background:color-mix(in srgb,var(--bg-card) 82%,transparent);
  color:var(--text-main);
  box-shadow:0 8px 24px #0005;
  opacity:.28;
  cursor:pointer;
  transition:opacity .15s ease,background .15s ease;
}
.room-focus-exit:hover,.room-focus-exit:focus-visible { opacity:1;background:var(--bg-card); }
.room-focus-exit:focus-visible { outline:2px solid var(--accent);outline-offset:2px; }
@supports not (color:color-mix(in srgb,white,black)) {
  .room-focus-exit { border-color:var(--border-main);background:var(--bg-card); }
}
</style>
