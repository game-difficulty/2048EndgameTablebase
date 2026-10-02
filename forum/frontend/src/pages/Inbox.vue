<script setup>
import { inject, ref, watch } from "vue";
import { api } from "../api";
const { user, notifications, notificationsConnected } = inject("forum"),
  items = ref([]),
  error = ref(""),
  loading = ref(false);
let epoch = 0;
async function load(more = false) {
  const t = ++epoch;
  if (!more) items.value = [];
  if (!user.value) return;
  loading.value = true;
  try {
    const r = await api(
      "/notifications" +
        (more && items.value.length ? "?before=" + items.value.at(-1).id : ""),
    );
    if (t === epoch)
      items.value = more ? [...items.value, ...r.items] : r.items;
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
watch(
  () => user.value?.id,
  () => load(),
  { immediate: true },
);
watch(
  () => notifications.value.latest,
  () => load(),
);
async function preference() {
  try {
    const r = await api("/notification-preferences", {
      method: "PUT",
      body: { enabled: !notifications.value.enabled },
    });
    notifications.value = { ...notifications.value, ...r };
  } catch (e) {
    error.value = e.message;
  }
}
</script>
<template>
  <div class="heading">
    <h1>社区通知</h1>
    <button v-if="items.length" @click="read">全部标为已读</button>
  </div>
  <div v-if="user" class="actions">
    <span class="muted">{{
      notificationsConnected ? "实时连接正常" : "正在重新连接，历史通知仍可查看"
    }}</span
    ><button @click="preference">
      {{ notifications.enabled ? "暂停新通知" : "开启新通知" }}</button
    ><button @click="load()">刷新</button>
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
      <p>
        {{ item.actor_name }}
        {{
          {
            mention: "提及了你",
            follow: "发布了新主题",
            subscription: "更新了你的订阅",
            reply: "回复了讨论",
          }[item.kind] || "回复了讨论"
        }}
      </p>
      <RouterLink
        v-if="item.available"
        :to="`/t/${item.topic_id}#p-${item.post_id}`"
        >{{ item.title }}</RouterLink
      ><span v-else class="muted">{{ item.title }}</span>
      <p class="muted">{{ new Date(item.created_at).toLocaleString() }}</p>
    </div>
  </article>
  <button v-if="items.length >= 100" :disabled="loading" @click="load(true)">
    加载更早通知
  </button>
</template>
