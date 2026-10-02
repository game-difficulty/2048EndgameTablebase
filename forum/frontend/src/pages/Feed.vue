<script setup>
import { computed, inject, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api";
const route = useRoute(),
  { boards, user } = inject("forum");
const items = ref([]),
  cursor = ref(""),
  loading = ref(false),
  error = ref(""),
  query = ref("");
let generation = 0;
const current = computed(() =>
  boards.value.find((b) => b.slug === route.params.slug),
);
const title = computed(() =>
  route.path === "/bookmarks"
    ? "我的收藏"
    : current.value?.name || "每一局，都值得聊聊。",
);
async function load(more = false) {
  const ticket = ++generation;
  loading.value = true;
  error.value = "";
  if (!more) {
    items.value = [];
    cursor.value = "";
  }
  const params = new URLSearchParams({
    board: route.params.slug || "",
    q: query.value,
    saved: String(route.path === "/bookmarks"),
  });
  if (more) params.set("cursor", cursor.value);
  try {
    const data = await api("/topics?" + params);
    if (ticket !== generation) return;
    items.value = more
      ? [
          ...items.value,
          ...data.items.filter((t) => !items.value.some((x) => x.id === t.id)),
        ]
      : data.items;
    cursor.value = data.next_cursor;
  } catch (e) {
    if (ticket === generation) error.value = e.message;
  } finally {
    if (ticket === generation) loading.value = false;
  }
}
function when(value) {
  return new Date(value).toLocaleString("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
watch(
  () => [route.path, user.value?.id],
  () => load(),
  { immediate: true },
);
</script>
<template>
  <div class="heading">
    <div>
      <p class="eyebrow">2048 COMMUNITY</p>
      <h1>{{ title }}</h1>
      <p class="muted">
        {{ current?.description || "分享你的发现，讨论有趣的局面。" }}
      </p>
    </div>
    <RouterLink
      v-if="user && (!current || current.can_post)"
      class="button primary"
      :to="{ path: '/compose', query: { board: current?.slug || 'general' } }"
      >＋ 发布主题</RouterLink
    >
  </div>
  <form class="search" @submit.prevent="load()">
    <label class="sr-only" for="topic-search">搜索主题与正文</label
    ><input
      id="topic-search"
      v-model="query"
      maxlength="80"
      placeholder="搜索讨论、复盘、32K…"
    /><button :disabled="loading">搜索</button>
  </form>
  <p class="notice">分享局面时，请说明来源、规则和你的思路。</p>
  <div v-if="error" class="notice error" role="alert">
    {{ error }} <button @click="load()">重试</button>
  </div>
  <div v-if="!items.length && !loading && !error" class="empty">
    <h2>这里还没有讨论</h2>
    <p>从一个问题、一段经历或一个棋盘开始。</p>
    <RouterLink v-if="user" to="/compose">写下第一篇主题</RouterLink>
  </div>
  <article v-for="topic in items" :key="topic.id" class="topic-row">
    <div>
      <RouterLink :to="'/t/' + topic.id" class="topic-title">{{
        topic.title
      }}</RouterLink>
      <div class="meta">
        <RouterLink :to="'/c/' + topic.board_slug" class="badge">{{
          topic.board_name
        }}</RouterLink
        ><span v-for="tag in topic.tags" :key="tag">#{{ tag }}</span
        ><span>{{ topic.display_name }}</span
        ><span>{{ when(topic.last_activity) }}</span
        ><span v-if="topic.locked">已锁定</span>
      </div>
    </div>
    <span class="reply-count">{{ topic.replies }} <small>回复</small></span>
  </article>
  <p v-if="loading" class="empty" role="status">正在加载…</p>
  <button v-else-if="cursor" class="load-more" @click="load(true)">
    加载更多
  </button>
</template>
