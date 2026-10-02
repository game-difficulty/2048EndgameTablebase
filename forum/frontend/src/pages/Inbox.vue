<script setup>
import { inject, ref, watch } from "vue";
import { api } from "../api";
const { user } = inject("forum"),
  items = ref([]),
  error = ref(""),
  loading = ref(false);
let epoch = 0;
async function load() {
  const t = ++epoch;
  items.value = [];
  if (!user.value) return;
  loading.value = true;
  try {
    const r = await api("/notifications");
    if (t === epoch) items.value = r.items;
  } catch (e) {
    error.value = e.message;
  } finally {
    if (t === epoch) loading.value = false;
  }
}
async function read() {
  try {
    await api(
      "/notifications/read?through_id=" +
        Math.max(...items.value.map((x) => x.id)),
      { method: "PUT" },
    );
    await load();
  } catch (e) {
    error.value = e.message;
  }
}
watch(() => user.value?.id, load, { immediate: true });
</script>
<template>
  <div class="heading">
    <h1>回复通知</h1>
    <button v-if="items.length" @click="read">全部标为已读</button>
  </div>
  <p v-if="error" class="notice error" role="alert">
    {{ error }} <button @click="load">重试</button>
  </p>
  <p v-if="!user" class="empty">登录后查看你的通知。</p>
  <p v-else-if="loading" class="empty">正在加载…</p>
  <p v-else-if="!items.length" class="empty">
    还没有新通知，收到回复后会显示在这里。
  </p>
  <article v-for="item in items" :key="item.id" class="topic-row">
    <div>
      <span v-if="!item.read_at" class="badge">未读</span>
      <p>{{ item.actor_name }} 回复了讨论</p>
      <RouterLink
        v-if="item.available"
        :to="`/t/${item.topic_id}#p-${item.post_id}`"
        >{{ item.title }}</RouterLink
      ><span v-else class="muted">{{ item.title }}</span>
      <p class="muted">{{ new Date(item.created_at).toLocaleString() }}</p>
    </div>
  </article>
</template>
