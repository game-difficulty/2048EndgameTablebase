<script setup>
import { computed, onMounted, onUnmounted, provide, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api } from "./api";
import { refreshTilePalette } from "./tilePalette";
import { useNotifications } from "./notifications";
const router = useRouter(),
  route = useRoute();
const globalQuery = ref("");
const pageClass = computed(() =>
  route.path.startsWith("/t/")
    ? "page-topic"
    : route.path === "/compose"
      ? "page-compose"
      : ["/moderation", "/operations"].includes(route.path)
        ? "page-admin"
        : "",
);
function search() {
  router.push({
    path: "/",
    query: globalQuery.value.trim() ? { q: globalQuery.value.trim() } : {},
  });
}
const session = ref(null),
  boards = ref([]),
  error = ref(""),
  loading = ref(true),
  theme = ref("system");
const user = computed(() => session.value?.user);
const { status: notifications, connected: notificationsConnected } =
  useNotifications(user);
let sessionRequest = 0;
provide("forum", {
  session,
  boards,
  user,
  refresh,
  notifications,
  notificationsConnected,
});
async function refresh() {
  const ticket = ++sessionRequest;
  error.value = "";
  loading.value = true;
  try {
    const [s, b] = await Promise.all([api("/session"), api("/boards")]);
    if (ticket !== sessionRequest) return;
    session.value = s;
    boards.value = b.items;
  } catch (e) {
    if (ticket === sessionRequest) error.value = e.message;
  } finally {
    if (ticket === sessionRequest) loading.value = false;
  }
}
function expired() {
  session.value = null;
  refresh();
}
function appearance() {
  refreshTilePalette();
  document.documentElement.dataset.theme = theme.value;
  try {
    localStorage.setItem("forum:theme", theme.value);
  } catch {}
}
function foreground() {
  if (!document.hidden) {
    refreshTilePalette();
    refresh();
  }
}
onMounted(() => {
  try {
    theme.value = localStorage.getItem("forum:theme") || "system";
  } catch {}
  appearance();
  refresh();
  window.addEventListener("forum-auth-expired", expired);
  document.addEventListener("visibilitychange", foreground);
  window.addEventListener("focus", refreshTilePalette);
});
onUnmounted(() => {
  window.removeEventListener("forum-auth-expired", expired);
  document.removeEventListener("visibilitychange", foreground);
  window.removeEventListener("focus", refreshTilePalette);
});
</script>
<template>
  <a class="skip-link" href="#main-content">跳到主要内容</a>
  <header class="topbar">
    <RouterLink to="/" class="brand"
      ><span class="brand-tile">2048</span
      ><span>社区<small>一起研究下一步</small></span></RouterLink
    >
    <nav aria-label="站点导航" class="site-links">
      <a href="https://2048tables.online/">主站</a
      ><a href="https://play.2048tables.online/">Play</a
      ><a href="https://live.2048tables.online/">直播</a
      ><a href="https://tournament.2048tables.online/">赛事</a>
    </nav>
    <form class="global-search" role="search" @submit.prevent="search">
      <label class="sr-only" for="global-search">搜索社区</label
      ><input
        id="global-search"
        v-model="globalQuery"
        maxlength="80"
        placeholder="搜索主题、局面与思路"
      /><button aria-label="搜索社区">搜索</button>
    </form>
    <div class="account">
      <label class="sr-only" for="appearance">外观</label
      ><select id="appearance" v-model="theme" @change="appearance">
        <option value="system">系统</option>
        <option value="light">浅色</option>
        <option value="dark">深色</option>
      </select>
      <RouterLink
        v-if="user"
        :to="'/u/' + user.id"
        class="account-link"
        :aria-label="user.display_name + ' 的个人页'"
        ><span class="avatar">{{ user.display_name.slice(0, 1) }}</span
        ><span>{{ user.display_name }}</span></RouterLink
      >
      <a v-else href="https://play.2048tables.online/">登录</a>
    </div>
  </header>
  <div v-if="session?.development_auth" class="dev-note">
    本地预览 · 测试身份 · 未连接正式论坛数据
  </div>
  <div v-if="error" class="notice error" role="alert">
    {{ error }} <button @click="refresh">重新连接</button>
  </div>
  <div class="shell" :class="pageClass">
    <aside class="sidebar" aria-label="社区导航">
      <div class="sidebar-inner">
        <RouterLink
          v-if="user"
          to="/compose"
          class="button primary sidebar-create"
          >＋ 发布主题</RouterLink
        >
        <p class="eyebrow">讨论空间</p>
        <nav class="side-links" aria-label="板块导航">
          <RouterLink to="/" :class="{ selected: route.path === '/' }"
            ><span class="nav-symbol" aria-hidden="true">▦</span
            >全部讨论</RouterLink
          >
          <RouterLink
            v-for="b in boards"
            :key="b.id"
            :to="'/c/' + b.slug"
            :class="{ selected: route.params.slug === b.slug }"
            ><span class="nav-symbol" aria-hidden="true">#</span
            >{{ b.name }}</RouterLink
          >
        </nav>
        <template v-if="user">
          <p class="eyebrow">个人空间</p>
          <nav class="side-links" aria-label="个人导航">
            <RouterLink to="/notifications"
              ><span class="nav-symbol" aria-hidden="true">◉</span>通知<span
                v-if="notifications.unread"
                class="nav-count"
                >{{
                  notifications.unread > 99 ? "99+" : notifications.unread
                }}</span
              ></RouterLink
            >
            <RouterLink to="/bookmarks"
              ><span class="nav-symbol" aria-hidden="true">◇</span
              >我的收藏</RouterLink
            >
            <RouterLink to="/community"
              ><span class="nav-symbol" aria-hidden="true">▤</span
              >我的社区</RouterLink
            >
            <RouterLink to="/settings"
              ><span class="nav-symbol" aria-hidden="true">⚙</span
              >偏好与隐私</RouterLink
            >
          </nav>
        </template>
        <template v-if="session?.can_moderate"
          ><p class="eyebrow">管理工作台</p>
          <nav class="side-links" aria-label="管理导航">
            <RouterLink to="/moderation"
              ><span class="nav-symbol" aria-hidden="true">▣</span
              >内容与审核</RouterLink
            ><RouterLink v-if="session?.is_admin" to="/operations"
              ><span class="nav-symbol" aria-hidden="true">◷</span
              >公告与运营</RouterLink
            >
          </nav></template
        >
        <div class="sidebar-note">
          <strong>分享高光，也分享思路。</strong>
          <p>一个局面、一次尝试，都可以成为讨论的开始。</p>
        </div>
      </div>
    </aside>
    <main id="main-content" tabindex="-1">
      <div class="mobile-nav">
        <label class="sr-only" for="board-nav">切换板块</label
        ><select
          id="board-nav"
          :value="route.params.slug || ''"
          @change="
            router.push($event.target.value ? '/c/' + $event.target.value : '/')
          "
        >
          <option value="">全部讨论</option>
          <option v-for="b in boards" :key="b.id" :value="b.slug">
            {{ b.name }}
          </option></select
        ><RouterLink v-if="session?.can_moderate" to="/moderation"
          >管理工作台</RouterLink
        >
      </div>
      <div v-if="loading && !session" class="empty" role="status">
        正在连接社区…
      </div>
      <RouterView v-else />
    </main>
  </div>
  <nav class="mobile-bottom" aria-label="快捷导航">
    <RouterLink to="/">讨论</RouterLink
    ><RouterLink v-if="user" to="/compose">＋ 创作</RouterLink
    ><RouterLink v-if="user" to="/notifications"
      >通知<span v-if="notifications.unread" class="nav-count">{{
        notifications.unread > 99 ? "99+" : notifications.unread
      }}</span></RouterLink
    ><RouterLink v-if="user" to="/community">我的</RouterLink
    ><a v-else href="https://play.2048tables.online/">登录社区</a>
  </nav>
  <footer>2048 社区 <span>·</span> 分享高光，也一起研究下一步。</footer>
</template>
