<script setup>
import { ref } from "vue";
import { api } from "../api";
const emit = defineEmits(["insert"]);
const query = ref(""),
  items = ref([]),
  error = ref(""),
  busy = ref(false);
async function search() {
  busy.value = true;
  error.value = "";
  try {
    items.value = (
      await api("/profiles?q=" + encodeURIComponent(query.value.trim()))
    ).items;
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
</script>
<template>
  <details class="mention-picker">
    <summary>提及社区用户</summary>
    <p class="muted">
      按昵称或用户 ID 搜索。每条内容最多提及 10 人；代码和转义文本不会通知用户。
    </p>
    <div class="actions">
      <label
        >昵称或 ID
        <input v-model="query" maxlength="80" @keydown.enter.prevent="search"
      /></label>
      <button type="button" :disabled="busy" @click="search">查找用户</button>
    </div>
    <p v-if="error" role="alert">{{ error }}</p>
    <div class="actions">
      <button
        v-for="item in items"
        :key="item.user_id"
        type="button"
        @click="emit('insert', `<@${item.user_id}>`)"
      >
        {{ item.display_name }} · #{{ item.user_id }}
      </button>
    </div>
  </details>
</template>
