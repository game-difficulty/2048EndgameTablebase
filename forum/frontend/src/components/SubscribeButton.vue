<script setup>
import { inject, ref, watch } from "vue";
import { api } from "../api";
const props = defineProps({ kind: String, id: Number });
const { user } = inject("forum"),
  active = ref(false),
  busy = ref(false),
  error = ref("");
let epoch = 0;
watch(
  [() => props.kind, () => props.id, () => user.value?.id],
  async () => {
    const ticket = ++epoch;
    active.value = false;
    error.value = "";
    if (!user.value || !props.id) return;
    try {
      const r = await api("/subscriptions");
      if (ticket === epoch)
        active.value = r.items.some(
          (s) => s.kind === props.kind && s.target_id === props.id,
        );
    } catch (e) {
      if (ticket === epoch) error.value = e.message;
    }
  },
  { immediate: true },
);
async function toggle() {
  busy.value = true;
  error.value = "";
  try {
    await api(`/subscriptions/${props.kind}/${props.id}`, {
      method: active.value ? "DELETE" : "PUT",
    });
    active.value = !active.value;
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
</script>
<template>
  <span v-if="user"
    ><button :disabled="busy" :aria-pressed="active" @click="toggle">
      {{
        active ? "取消订阅" : kind === "board" ? "订阅新主题" : "订阅新回复"
      }}</button
    ><small v-if="error" role="alert">{{ error }}</small></span
  >
</template>
