<script setup>
import { computed, inject, ref, watch } from "vue";
import { groupNotifications } from "../notificationGroups";
import { api } from "../api";
const { user, notifications, notificationsConnected } = inject("forum"),
  items = ref([]),
  error = ref(""),
  loading = ref(false);
const kind = ref("");
const groups = computed(() => groupNotifications(items.value));
let epoch = 0;
async function load(more = false) {
  const t = ++epoch;
  if (!more) items.value = [];
  if (!user.value) return;
  loading.value = true;
  try {
    const r = await api(
      "/notifications?kind=" +
        kind.value +
        (more && items.value.length ? "&before=" + items.value.at(-1).id : ""),
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
watch(kind, () => load());
</script>
<template>
  <div class="heading">
    <div>
      <p class="eyebrow">个人空间 / 消息</p>
      <h1>社区通知</h1>
      <p class="muted">回复、提及，还有你关心的讨论。</p>
    </div>
    <button v-if="items.length" @click="read">全部标为已读</button>
  </div>
  <div v-if="user" class="actions">
    <RouterLink to="/settings">通知设置</RouterLink
    ><label
      >通知类型<select v-model="kind">
        <option value="">全部</option>
        <option value="reply">回复</option>
        <option value="mention">提及</option>
        <option value="subscription">订阅</option>
        <option value="follow">关注</option>
        <option value="moderation">管理与申诉</option>
        <option value="system">系统</option>
      </select></label
    >
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
  <article
    v-for="item in groups"
    :key="item.id"
    class="topic-row inbox-row"
    :class="{ unread: !item.read_at }"
  >
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
            moderation: "处理结果",
            system: "系统消息",
          }[item.kind] || "回复了讨论"
        }}
      </p>
      <RouterLink
        v-if="item.available"
        :to="item.path || `/t/${item.topic_id}#p-${item.post_id}`"
        >{{ item.title }}</RouterLink
      ><span v-else class="muted">{{ item.title }}</span>
      <p v-if="item.body">{{ item.body }}</p>
      <details v-if="item.notices.length > 1">
        <summary>同主题 {{ item.notices.length }} 条通知（本次已加载）</summary>
        <p v-for="notice in item.notices" :key="notice.id">
          {{ notice.actor_name }} ·
          <RouterLink v-if="notice.available" :to="notice.path"
            >查看 #{{ notice.post_number }}</RouterLink
          >
          <span v-else>内容已不可见</span>
        </p>
      </details>
      <p class="muted">{{ new Date(item.created_at).toLocaleString() }}</p>
    </div>
  </article>
  <button v-if="items.length >= 100" :disabled="loading" @click="load(true)">
    加载更早通知
  </button>
</template>
