<script setup>
import { inject, ref, watch } from "vue";
import { api } from "../api";
const { user, session } = inject("forum"),
  items = ref([]),
  error = ref(""),
  loading = ref(false);
let epoch = 0;
async function load() {
  const t = ++epoch;
  items.value = [];
  if (!session.value?.can_moderate) return;
  loading.value = true;
  try {
    const result = await api("/moderation/reports");
    if (t === epoch) items.value = result.items;
  } catch (e) {
    error.value = e.message;
  } finally {
    if (t === epoch) loading.value = false;
  }
}
watch(() => [user.value?.id, session.value?.can_moderate], load, {
  immediate: true,
});
</script>
<template>
  <div class="heading">
    <div>
      <p class="eyebrow">MODERATION</p>
      <h1>举报审核队列</h1>
    </div>
    <button @click="load">刷新</button>
  </div>
  <p v-if="error" class="notice error" role="alert">{{ error }}</p>
  <p v-if="!session?.can_moderate" class="empty">需要相应板块的管理权限。</p>
  <p v-else-if="loading" class="empty">正在读取举报…</p>
  <p v-else-if="!items.length" class="empty">当前没有待处理举报。</p>
  <article v-for="report in items" :key="report.id" class="panel">
    <RouterLink :to="`/t/${report.topic_id}#p-${report.post_id}`"
      >{{ report.title }} · #{{ report.post_number }}</RouterLink
    >
    <p class="prose">{{ report.reason }}</p>
    <p class="muted">
      举报 #{{ report.id }} · {{ new Date(report.created_at).toLocaleString() }}
    </p>
    <p class="muted">打开讨论查看上下文，并填写处置原因。</p>
  </article>
</template>
