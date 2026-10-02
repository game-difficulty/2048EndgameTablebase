<script setup>
import { computed, onMounted, onUnmounted, provide, ref } from "vue";
import { useRouter } from "vue-router";
import { api } from "./api";
import { refreshTilePalette } from "./tilePalette";
import { useNotifications } from "./notifications";
const router = useRouter();
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
  <header class="topbar">
    <RouterLink to="/" class="brand"><strong>2048</strong> 社区</RouterLink>
    <nav aria-label="站点导航">
      <a href="https://2048tables.online/">主站</a
      ><a href="https://play.2048tables.online/">Play</a
      ><a href="https://live.2048tables.online/">直播</a
      ><a href="https://tournament.2048tables.online/">赛事</a
      ><RouterLink to="/">社区</RouterLink>
    </nav>
    <div class="account">
      <label class="sr-only" for="appearance">外观</label
      ><select id="appearance" v-model="theme" @change="appearance">
        <option value="system">跟随系统</option>
        <option value="light">浅色</option>
        <option value="dark">深色</option></select
      ><span v-if="user">{{ user.display_name }}</span
      ><a v-else href="https://play.2048tables.online/">前往 Play 登录</a>
    </div>
  </header>
  <div v-if="session?.development_auth" class="dev-note">
    本地开发环境 · 使用测试身份，未连接正式论坛数据
  </div>
  <div v-if="error" class="notice error" role="alert">
    {{ error }} <button @click="refresh">重新连接</button>
  </div>
  <div class="shell">
    <aside class="sidebar">
      <p class="eyebrow">发现讨论</p>
      <RouterLink to="/" :class="{ selected: $route.path === '/' }"
        >全部讨论</RouterLink
      ><RouterLink
        v-for="b in boards"
        :key="b.id"
        :to="'/c/' + b.slug"
        :class="{ selected: $route.params.slug === b.slug }"
        >{{ b.name }}</RouterLink
      ><template v-if="user"
        ><p class="eyebrow">我的社区</p>
        <RouterLink to="/bookmarks">我的收藏</RouterLink
        ><RouterLink to="/notifications"
          >通知 {{ notifications.unread || "" }}</RouterLink
        ><RouterLink to="/community">我的社区</RouterLink
        ><RouterLink to="/compose">草稿与创作</RouterLink
        ><RouterLink v-if="session.can_moderate" to="/moderation"
          >审核队列</RouterLink
        ></template
      >
    </aside>
    <main>
      <div class="mobile-nav">
        <label class="sr-only" for="board-nav">板块</label
        ><select
          id="board-nav"
          :value="$route.params.slug || ''"
          @change="
            router.push($event.target.value ? '/c/' + $event.target.value : '/')
          "
        >
          <option value="">全部讨论</option>
          <option v-for="b in boards" :value="b.slug" :key="b.id">
            {{ b.name }}
          </option></select
        ><RouterLink v-if="user" to="/notifications"
          >通知 {{ notifications.unread || "" }}</RouterLink
        ><RouterLink v-if="user" to="/community">我的</RouterLink
        ><RouterLink v-if="user" to="/bookmarks">收藏</RouterLink
        ><RouterLink v-if="session?.can_moderate" to="/moderation"
          >审核</RouterLink
        >
      </div>
      <div v-if="loading && !session" class="empty" role="status">
        正在连接社区…
      </div>
      <RouterView v-else />
    </main>
  </div>
  <footer>2048 社区 · 分享高光，也一起研究下一步。</footer>
</template>
