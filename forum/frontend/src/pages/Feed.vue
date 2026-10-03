<script setup>
import { computed, inject, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api } from "../api";
import SubscribeButton from "../components/SubscribeButton.vue";
const router = useRouter();
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
  try {
    if (since.value) params.set("since", new Date(since.value).toISOString());
    if (until.value) params.set("until", new Date(until.value).toISOString());
    if (more) params.set("cursor", cursor.value);
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
const views = computed(() => [
  ["activity", "最新回复"],
  ["newest", "最新发布"],
  ["featured", "精选"],
  ["unanswered", "未回复"],
  ...(user.value
    ? [
        ["following", "关注"],
        ["unread", "未读"],
      ]
    : []),
]);
const filtered = computed(
  () =>
    !!(
      query.value ||
      tag.value ||
      author.value ||
      topicId.value ||
      kind.value ||
      since.value ||
      until.value
    ),
);
function applyFilters() {
  const q = {};
  for (const [k, v] of Object.entries({
    q: query.value,
    view: view.value,
    tag: tag.value,
    author: author.value,
    topic: topicId.value,
    kind: kind.value,
    since: since.value,
    until: until.value,
  }))
    if (v && (k !== "view" || v !== "activity")) q[k] = v;
  if (JSON.stringify(q) === JSON.stringify(route.query)) load();
  else router.replace({ path: route.path, query: q });
}
function chooseView(value) {
  view.value = value;
  applyFilters();
}
function clearFilters() {
  query.value =
    tag.value =
    author.value =
    topicId.value =
    kind.value =
    since.value =
    until.value =
      "";
  applyFilters();
}
watch(
  [() => route.fullPath, () => user.value?.id],
  () => {
    const value = (k) =>
      typeof route.query[k] === "string" ? route.query[k] : "";
    query.value = value("q").slice(0, 80);
    tag.value = value("tag").slice(0, 24);
    author.value = value("author");
    topicId.value = value("topic");
    kind.value = ["discussion", "question", "poll"].includes(value("kind"))
      ? value("kind")
      : "";
    since.value = value("since");
    until.value = value("until");
    view.value = views.value.some(([key]) => key === value("view"))
      ? value("view")
      : "activity";
    load();
  },
  { immediate: true },
);
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
  <header class="heading feed-heading">
    <div>
      <p class="eyebrow">
        讨论 /
        {{
          current?.name ||
          (route.path === "/bookmarks" ? "我的收藏" : "社区首页")
        }}
      </p>
      <h1>{{ title }}</h1>
      <p class="muted">
        {{
          current?.description ||
          (route.path === "/bookmarks"
            ? "把值得再读的思路，留在这里。"
            : "分享高分与复盘，交流攻略，也聊聊下一步。")
        }}
      </p>
    </div>
    <SubscribeButton v-if="current" kind="board" :id="current.id" />
  </header>
  <div class="feed-layout">
    <section class="discussion-column" aria-label="讨论列表">
      <div v-if="announcements.length" class="announcement-strip">
        <span class="badge badge-gold">公告</span
        ><RouterLink :to="'/t/' + announcements[0].topic_id">{{
          announcements[0].title
        }}</RouterLink
        ><span aria-hidden="true">↗</span>
      </div>
      <div class="feed-surface">
        <nav class="view-tabs" aria-label="讨论视图">
          <button
            v-for="[key, label] in views"
            :key="key"
            :aria-pressed="view === key"
            @click="chooseView(key)"
          >
            {{ label }}
          </button>
        </nav>
        <form
          class="search feed-search"
          role="search"
          @submit.prevent="applyFilters"
        >
          <label class="sr-only" for="topic-search">搜索主题与正文</label
          ><input
            id="topic-search"
            v-model="query"
            maxlength="80"
            placeholder="搜索当前讨论…"
          /><button :disabled="loading">搜索</button>
        </form>
        <details class="feed-filters">
          <summary>
            筛选讨论<span v-if="filtered" class="badge">已设置</span>
          </summary>
          <div class="filter-grid">
            <label class="field"
              >标签<input
                v-model="tag"
                maxlength="24"
                placeholder="例如 32K" /></label
            ><label class="field"
              >作者 ID<input v-model="author" type="number" min="1" /></label
            ><label class="field"
              >主题 ID<input v-model="topicId" type="number" min="1" /></label
            ><label class="field"
              >内容类型<select v-model="kind">
                <option value="">全部</option>
                <option value="discussion">讨论</option>
                <option value="question">问答</option>
                <option value="poll">投票</option>
              </select></label
            ><label class="field"
              >发布于<input v-model="since" type="datetime-local" /></label
            ><label class="field"
              >截至<input v-model="until" type="datetime-local"
            /></label>
          </div>
          <div class="actions">
            <button :disabled="loading" @click="applyFilters">应用筛选</button
            ><button v-if="filtered" @click="clearFilters">清除筛选</button>
          </div>
        </details>
        <div class="list-caption">
          <span>{{ filtered ? "筛选结果" : "正在讨论" }}</span
          ><span>回复</span>
        </div>
        <div v-if="error" class="notice error" role="alert">
          {{ error }} <button @click="load()">重试</button>
        </div>
        <div v-if="!items.length && !loading && !error" class="empty">
          <span class="empty-symbol" aria-hidden="true">#</span>
          <h2>{{ filtered ? "还没有匹配的讨论" : "从你的第一句话开始" }}</h2>
          <p>
            {{
              filtered
                ? "换一个关键词，或放宽筛选条件。"
                : "一个问题、一段复盘、一个棋盘，都值得分享。"
            }}
          </p>
          <button v-if="filtered" @click="clearFilters">清除筛选</button
          ><RouterLink
            v-else-if="user && (!current || current.can_post)"
            class="button primary"
            :to="{
              path: '/compose',
              query: { board: current?.slug || 'general' },
            }"
            >发布主题</RouterLink
          >
        </div>
        <article
          v-for="topic in items"
          :key="topic.id"
          class="topic-row feed-topic"
        >
          <RouterLink
            :to="'/u/' + topic.author_id"
            class="avatar topic-avatar"
            :aria-label="'查看 ' + topic.display_name + ' 的个人页'"
            >{{ topic.display_name.slice(0, 1) }}</RouterLink
          >
          <div class="topic-row-main">
            <div class="topic-flags">
              <span v-if="topic.pinned" class="badge badge-gold">置顶</span
              ><span
                v-if="topic.kind === 'question'"
                class="badge"
                :class="{ 'badge-solved': topic.question_status === 'solved' }"
                >{{
                  { open: "问答", solved: "已解决", closed: "已关闭" }[
                    topic.question_status
                  ]
                }}</span
              ><span v-if="topic.kind === 'poll'" class="badge">投票</span>
            </div>
            <RouterLink :to="'/t/' + topic.id" class="topic-title">{{
              topic.title
            }}</RouterLink>
            <div class="meta">
              <RouterLink :to="'/c/' + topic.board_slug">{{
                topic.board_name
              }}</RouterLink
              ><RouterLink :to="'/u/' + topic.author_id">{{
                topic.display_name
              }}</RouterLink
              ><span>{{ when(topic.last_activity) }}</span
              ><span v-if="topic.locked">已锁定</span>
            </div>
            <div v-if="topic.tags.length" class="topic-tags">
              <button
                v-for="t in topic.tags"
                :key="t"
                @click="
                  tag = t;
                  applyFilters();
                "
              >
                # {{ t }}
              </button>
            </div>
          </div>
          <RouterLink
            :to="'/t/' + topic.id"
            class="reply-count"
            :aria-label="topic.replies + ' 条回复'"
            >{{ topic.replies }}</RouterLink
          >
        </article>
        <p v-if="loading" class="empty" role="status">正在加载讨论…</p>
        <button v-else-if="cursor" class="load-more" @click="load(true)">
          加载更多讨论 ↓
        </button>
        <p v-else-if="items.length" class="list-end">已经看到这里的全部讨论</p>
      </div>
    </section>
    <aside class="context-rail" aria-label="社区指南与动态">
      <section class="rail-card welcome-card">
        <p class="eyebrow">你的下一篇分享</p>
        <h2>不只晒分数，<br />也聊聊怎么做到。</h2>
        <p>复盘关键一步，提出一个问题，或发起一次投票。</p>
        <RouterLink
          v-if="user && (!current || current.can_post)"
          class="button primary"
          :to="{
            path: '/compose',
            query: { board: current?.slug || 'general' },
          }"
          >＋ 开始创作</RouterLink
        ><a
          v-else-if="!user"
          class="button primary"
          href="https://play.2048tables.online/"
          >登录后参与</a
        >
      </section>
      <section class="rail-card">
        <h2>用文字分享局面</h2>
        <p class="muted">输入变体与盘面编码，发布后自动显示棋盘。</p>
        <code class="syntax-example"
          >[[board:4x4:<br />fedc/ba98/7654/3210]]</code
        >
        <p class="muted">还可以上传录像，从某一步继续讨论。</p>
        <RouterLink v-if="user" to="/compose">打开创作台 →</RouterLink>
      </section>
      <section v-if="announcements.length" class="rail-card">
        <h2>社区公告</h2>
        <RouterLink
          v-for="a in announcements"
          :key="a.id"
          class="rail-link"
          :to="'/t/' + a.topic_id"
          >{{ a.title }} <span>↗</span></RouterLink
        >
      </section>
      <section v-if="cards.length" class="rail-card">
        <h2>赛事与直播</h2>
        <article v-for="c in cards" :key="c.id" class="rail-item">
          <span class="eyebrow">{{
            c.source === "competition" ? "赛事动态" : "直播动态"
          }}</span
          ><a :href="cardUrl(c)" target="_blank" rel="noopener noreferrer"
            >{{ c.title }} ↗</a
          >
          <p class="muted">{{ c.summary }}</p>
          <RouterLink v-if="c.topic_id" :to="'/t/' + c.topic_id"
            >进入讨论 →</RouterLink
          >
        </article>
      </section>
      <p class="rail-footnote">
        分享时说明来源、规则与思路。<br />让每一份讨论都有据可循。
      </p>
    </aside>
  </div>
</template>
