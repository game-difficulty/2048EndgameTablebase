import { createApp, h } from "vue";
import { createRouter, createWebHistory } from "vue-router";
import App from "./App.vue";
import Feed from "./pages/Feed.vue";
import Topic from "./pages/Topic.vue";
import Compose from "./pages/Compose.vue";
import Inbox from "./pages/Inbox.vue";
import Moderation from "./pages/Moderation.vue";
import MyCommunity from "./pages/MyCommunity.vue";
import Profile from "./pages/Profile.vue";
import Settings from "./pages/Settings.vue";
import Operations from "./pages/Operations.vue";
import "./style.css";
import { refreshTilePalette } from "./tilePalette";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", component: Feed },
    { path: "/c/:slug", component: Feed },
    { path: "/bookmarks", component: Feed },
    { path: "/t/:id", component: Topic },
    { path: "/compose", component: Compose },
    { path: "/notifications", component: Inbox },
    { path: "/moderation", component: Moderation },
    { path: "/community", component: MyCommunity },
    { path: "/u/:id", component: Profile },
    { path: "/settings", component: Settings },
    { path: "/operations", component: Operations },
    {
      path: "/:pathMatch(.*)*",
      component: {
        render: () =>
          h("section", { class: "empty" }, [
            h("h1", "页面不存在"),
            h("a", { href: "/" }, "返回社区"),
          ]),
      },
    },
  ],
  scrollBehavior(to, from, saved) {
    // Topic fetches the anchored page and scrolls after its posts have rendered.
    if (to.hash && to.path.startsWith("/t/")) return false;
    return saved || { top: 0 };
  },
});
refreshTilePalette();
createApp(App).use(router).mount("#app");
