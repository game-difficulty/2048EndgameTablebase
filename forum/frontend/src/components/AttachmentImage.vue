<script setup>
import { ref, watch, onBeforeUnmount } from "vue";
import { mediaBlob } from "../api";
const props = defineProps({ id: String, alt: String });
const url = ref(""),
  error = ref("");
let controller;
function clear() {
  controller?.abort();
  if (url.value) URL.revokeObjectURL(url.value);
  url.value = "";
}
watch(
  () => props.id,
  async (id) => {
    clear();
    error.value = "";
    const current = new AbortController();
    controller = current;
    try {
      const blob = await mediaBlob(id, current.signal);
      if (!current.signal.aborted) url.value = URL.createObjectURL(blob);
    } catch (e) {
      if (!current.signal.aborted) error.value = e.message;
    }
  },
  { immediate: true },
);
onBeforeUnmount(clear);
</script>
<template>
  <span class="attachment-image"
    ><img v-if="url" :src="url" :alt="alt || '社区图片'" loading="lazy" /><span
      v-else
      class="muted"
      >{{ error || "正在加载图片…" }}</span
    ></span
  >
</template>
