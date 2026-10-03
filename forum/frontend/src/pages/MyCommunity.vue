<script setup>
import { inject, ref, watch } from "vue";
import { api } from "../api";
import MyAppeals from "../components/MyAppeals.vue";
const section = ref("reading");
const sections = [
  ["reading", "继续阅读"],
  ["drafts", "回复草稿"],
  ["following", "关注与订阅"],
  ["media", "我的附件"],
  ["appeals", "申诉记录"],
];
const { user } = inject("forum"),
  subscriptions = ref([]),
  media = ref([]),
  drafts = ref([]),
  follows = ref([]),
  reading = ref([]),
  error = ref(""),
  busy = ref(false);
let epoch = 0;
async function load() {
  const ticket = ++epoch;
  subscriptions.value = [];
  media.value = [];
  drafts.value = [];
  follows.value = [];
  reading.value = [];
  if (!user.value) return;
  try {
    const [s, m, f, r, d] = await Promise.all([
      api("/subscriptions"),
      api("/media"),
      api("/follows"),
      api("/reading"),
      api("/reply-drafts"),
    ]);
    if (ticket === epoch) {
      subscriptions.value = s.items;
      media.value = m.items;
      follows.value = f.items;
      reading.value = r.items;
      drafts.value = d.items;
    }
  } catch (e) {
    if (ticket === epoch) error.value = e.message;
  }
}
async function remove(path, body) {
  busy.value = true;
  error.value = "";
  try {
    await api(path, { method: "DELETE", body });
    await load();
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
function removeAsset(item) {
  if (confirm("移除此附件？所有引用它的帖子都将无法继续显示它。"))
    remove("/media/" + item.id, { reason: "用户主动撤回附件" });
}
async function removeDraft(item) {
  if (!confirm("清空这份回复草稿？")) return;
  await remove(`/reply-drafts/${item.topic_id}?revision=${item.revision}`);
  if (!error.value) {
    try {
      localStorage.removeItem(`forum.reply.${user.value.id}.${item.topic_id}`);
    } catch {
      /* optional cache */
    }
  }
}
watch(() => user.value?.id, load, { immediate: true });
</script>
<template>
  <div class="heading">
    <div>
      <p class="eyebrow">个人空间</p>
      <h1>我的社区</h1>
      <p class="muted">接着上次的讨论，继续还没写完的思路。</p>
    </div>
    <div class="actions">
      <RouterLink to="/bookmarks" class="button">我的收藏</RouterLink>
      <RouterLink to="/settings" class="button">偏好与隐私</RouterLink>
    </div>
  </div>
  <nav class="view-tabs" aria-label="我的社区功能">
    <button
      v-for="[key, label] in sections"
      :key="key"
      :aria-pressed="section === key"
      @click="section = key"
    >
      {{ label }}
    </button>
  </nav>
  <p v-if="error" class="notice error" role="alert">{{ error }}</p>
  <p v-if="!user" class="empty">请先登录。</p>
  <template v-else>
    <section v-if="section === 'drafts'" class="workspace-section">
      <div class="actions">
        <RouterLink class="button" to="/compose">主题草稿与创作 →</RouterLink>
      </div>
      <h2>回复草稿（{{ drafts.length }} / 50）</h2>
      <p v-if="!drafts.length" class="muted">
        尚未发布的回复会自动保存到这里。
      </p>
      <article v-for="d in drafts" :key="d.topic_id" class="panel">
        <RouterLink v-if="d.available" :to="'/t/' + d.topic_id">{{
          d.title
        }}</RouterLink
        ><strong v-else>{{ d.title }}</strong>
        <p>{{ d.preview }}</p>
        <button :disabled="busy" @click="removeDraft(d)">清空回复草稿</button>
      </article>
    </section>
    <section v-if="section === 'reading'" class="workspace-section">
      <h2>继续阅读</h2>
      <p v-if="!reading.length" class="muted">
        阅读到楼层末尾后，自动记住位置。
      </p>
      <article v-for="r in reading" :key="r.topic_id" class="panel">
        <RouterLink :to="`/t/${r.topic_id}#p-${r.post_id}`">{{
          r.title
        }}</RouterLink
        ><span class="muted"> · 上次到 #{{ r.post_number }}</span>
      </article>
    </section>
    <section v-if="section === 'following'" class="workspace-section">
      <h2>关注的用户（{{ follows.length }} / 200）</h2>
      <p v-if="!follows.length" class="muted">
        点击帖子作者进入个人页，即可关注。
      </p>
      <article v-for="f in follows" :key="f.followed_id" class="panel">
        <RouterLink :to="'/u/' + f.followed_id">{{ f.display_name }}</RouterLink
        ><button :disabled="busy" @click="remove('/follows/' + f.followed_id)">
          取消关注
        </button>
      </article>
      <h2>我的订阅</h2>
      <p class="muted">主题订阅通知新回复；板块订阅通知新主题。</p>
      <p v-if="!subscriptions.length">暂无订阅。</p>
      <article
        v-for="s in subscriptions"
        :key="s.kind + s.target_id"
        class="panel"
      >
        <RouterLink :to="s.path">{{ s.title }}</RouterLink>
        <button
          :disabled="busy"
          @click="remove(`/subscriptions/${s.kind}/${s.target_id}`)"
        >
          取消订阅
        </button>
      </article>
    </section>
    <section v-if="section === 'media'" class="workspace-section">
      <h2>我的附件（{{ media.length }} / 100）</h2>
      <p class="muted">
        容量上限 50
        MiB。未发布的附件仅自己可见；可移除放弃的草稿附件以释放配额。
      </p>
      <p v-if="!media.length">暂无附件。</p>
      <article v-for="m in media" :key="m.id" class="panel">
        <strong>{{
          m.kind === "image"
            ? "图片"
            : m.kind === "play"
              ? "Play 来源"
              : "上传录像"
        }}</strong>
        <p class="muted">
          {{ m.id }} · {{ (m.size / 1024).toFixed(1) }} KiB ·
          {{ new Date(m.created_at).toLocaleString() }}
        </p>
        <button :disabled="busy" @click="removeAsset(m)">移除附件</button>
      </article>
    </section>
    <MyAppeals v-if="section === 'appeals'" />
  </template>
</template>
