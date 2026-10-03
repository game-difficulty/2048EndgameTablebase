<script setup>
import { computed, inject, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api";
import SubscribeButton from "../components/SubscribeButton.vue";
const route = useRoute(),
  { boards, user } = inject("forum");
const items = ref([]),
  cursor = ref(""),
  loading = ref(false),
  error = ref(""),
  query = ref("");
const view = ref("activity"),
  tag = ref(""),
  author = ref(""),
  topicId = ref(""),
  kind = ref(""),
  since = ref(""),
  until = ref(""),
  announcements = ref([]),
  cards = ref([]);
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
    view: view.value,
    tag: tag.value,
    kind: kind.value,
  });
  if (author.value) params.set("author", author.value);
  if (topicId.value) params.set("topic_id", topicId.value);
  if (since.value) params.set("since", new Date(since.value).toISOString());
  if (until.value) params.set("until", new Date(until.value).toISOString());
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
watch([() => route.path, () => user.value?.id], () => load(), {
  immediate: true,
});
watch(view, () => load());
Promise.all([api("/announcements"), api("/external-cards")])
  .then(([a, c]) => {
    announcements.value = a.items;
    cards.value = c.items;
  })
  .catch(() => {});
function cardUrl(c) {
  return (
    (c.source === "competition"
      ? "https://tournament.2048tables.online"
      : "https://live.2048tables.online") + c.path
  );
}
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
  <SubscribeButton v-if="current" kind="board" :id="current.id" />
  <div v-for="a in announcements" :key="a.id" class="notice">
    <span class="badge">官方公告</span>
    <RouterLink :to="'/t/' + a.topic_id">{{ a.title }}</RouterLink>
  </div>
  <label class="field"
    >讨论视图<select v-model="view">
      <option value="activity">最新回复</option>
      <option value="newest">最新发布</option>
      <option value="featured">精选置顶</option>
      <option value="unanswered">未回复</option>
      <option v-if="user" value="following">关注作者</option>
      <option v-if="user" value="unread">未读主题</option>
    </select></label
  >
  <form class="search" @submit.prevent="load()">
    <label class="sr-only" for="topic-search">搜索主题与正文</label
    ><input
      id="topic-search"
      v-model="query"
      maxlength="80"
      placeholder="搜索讨论、复盘、32K…"
    /><button :disabled="loading">搜索</button>
  </form>
  <details class="panel">
    <summary>高级筛选</summary>
    <div class="filter-grid">
      <label class="field">精确标签<input v-model="tag" maxlength="24" /></label
      ><label class="field"
        >作者 ID<input v-model="author" type="number" min="1" /></label
      ><label class="field"
        >主题 ID<input v-model="topicId" type="number" min="1" /></label
      ><label class="field"
        >类型<select v-model="kind">
          <option value="">全部</option>
          <option value="discussion">讨论</option>
          <option value="question">问答</option>
          <option value="poll">投票</option>
        </select></label
      ><label class="field"
        >发布时间从<input v-model="since" type="datetime-local" /></label
      ><label class="field"
        >发布时间至<input v-model="until" type="datetime-local"
      /></label>
    </div>
    <button :disabled="loading" @click="load()">应用筛选</button>
  </details>
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
      <span v-if="topic.pinned" class="badge">置顶</span>
      <span v-if="topic.kind === 'question'" class="badge">{{
        topic.question_status === "solved" ? "已解决" : "提问"
      }}</span
      ><span v-if="topic.kind === 'poll'" class="badge">投票</span>
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
  <section v-if="cards.length" class="panel">
    <h2>赛事与直播动态</h2>
    <article v-for="c in cards" :key="c.id">
      <a :href="cardUrl(c)" target="_blank" rel="noopener noreferrer">{{
        c.title
      }}</a>
      <p>{{ c.summary }}</p>
      <p class="muted">
        {{ c.source === "competition" ? "赛事" : "直播" }} · 来源版本
        {{ c.revision }}
      </p>
      <RouterLink v-if="c.topic_id" :to="'/t/' + c.topic_id"
        >进入讨论</RouterLink
      >
    </article>
  </section>
</template>
