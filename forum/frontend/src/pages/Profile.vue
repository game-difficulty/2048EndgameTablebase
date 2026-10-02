<script setup>
import { inject, ref, watch } from "vue";
import { useRoute } from "vue-router";
import { api } from "../api";
const route = useRoute(),
  { user } = inject("forum");
const data = ref(null),
  error = ref(""),
  busy = ref(false),
  copied = ref(false);
let epoch = 0;
async function load(more = false) {
  const ticket = ++epoch;
  error.value = "";
  if (!more) data.value = null;
  try {
    const result = await api(
      `/profiles/${route.params.id}${more ? "?before=" + data.value.topics.at(-1).id : ""}`,
    );
    if (ticket === epoch)
      data.value = more
        ? { ...result, topics: [...data.value.topics, ...result.topics] }
        : result;
  } catch (e) {
    if (ticket === epoch) error.value = e.message;
  }
}
async function follow() {
  if (busy.value || !data.value) return;
  const target = data.value;
  busy.value = true;
  try {
    const r = await api("/follows/" + target.user_id, {
      method: target.following ? "DELETE" : "PUT",
    });
    if (data.value === target) data.value.following = r.following;
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
async function copy() {
  try {
    await navigator.clipboard.writeText(`<@${data.value.user_id}>`);
    copied.value = true;
  } catch {
    error.value = "无法访问剪贴板，请手动复制下方语法。";
  }
}
watch(
  [() => route.params.id, () => user.value?.id],
  () => {
    copied.value = false;
    load();
  },
  { immediate: true },
);
</script>
<template>
  <RouterLink to="/" class="back">← 返回讨论</RouterLink>
  <p v-if="error" class="notice error" role="alert">{{ error }}</p>
  <template v-if="data">
    <header class="profile-heading panel">
      <span class="avatar">{{ data.display_name.slice(0, 1) }}</span>
      <h1>{{ data.display_name }}</h1>
      <p class="muted">社区用户 #{{ data.user_id }}</p>
      <div class="actions">
        <button
          v-if="user && user.id !== data.user_id"
          class="primary"
          :disabled="busy"
          :aria-pressed="data.following"
          @click="follow"
        >
          {{ data.following ? "取消关注" : "关注用户" }}</button
        ><button @click="copy">
          {{ copied ? "提及语法已复制" : "复制提及语法" }}</button
        ><code>&lt;@{{ data.user_id }}&gt;</code>
      </div>
      <p class="muted">关注后，在对方发布新主题时收到通知。</p>
    </header>
    <h2>公开主题</h2>
    <p v-if="!data.topics.length" class="empty">暂无公开主题。</p>
    <article v-for="t in data.topics" :key="t.id" class="topic-row">
      <RouterLink :to="'/t/' + t.id">{{ t.title }}</RouterLink
      ><time class="muted">{{
        new Date(t.created_at).toLocaleDateString()
      }}</time>
    </article>
    <button
      v-if="data.topics.length && data.topics.length % 20 === 0"
      @click="load(true)"
    >
      加载更早主题
    </button>
  </template>
</template>
