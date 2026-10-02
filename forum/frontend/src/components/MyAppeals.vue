<script setup>
import { inject, ref, watch } from "vue";
import { api } from "../api";
const { user } = inject("forum");
const actions = ref([]),
  appeals = ref([]),
  selected = ref(null),
  reason = ref(""),
  error = ref(""),
  busy = ref(false);
const names = { hide: "隐藏主题", lock: "锁定主题", mute: "账号禁言" },
  states = { open: "等待处理", accepted: "申诉通过", rejected: "已结案" };
let epoch = 0;
async function load(more = false) {
  const ticket = ++epoch;
  if (!user.value) {
    actions.value = [];
    appeals.value = [];
    return;
  }
  error.value = "";
  try {
    const [a, b] = await Promise.all([
      api("/appeal-actions"),
      api("/appeals" + (more ? "?before=" + appeals.value.at(-1).id : "")),
    ]);
    if (ticket === epoch) {
      actions.value = a.items;
      appeals.value = more ? [...appeals.value, ...b.items] : b.items;
    }
  } catch (e) {
    if (ticket === epoch) error.value = e.message;
  }
}
async function submit() {
  busy.value = true;
  try {
    await api("/appeals", {
      method: "POST",
      body: { action_id: selected.value.id, reason: reason.value.trim() },
    });
    selected.value = null;
    reason.value = "";
    await load();
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
watch(
  () => user.value?.id,
  () => {
    selected.value = null;
    actions.value = [];
    appeals.value = [];
    load();
  },
  { immediate: true },
);
</script>
<template>
  <section v-if="user" class="appeals-section">
    <div class="heading">
      <h2>管理记录与申诉</h2>
      <button :disabled="busy" @click="load()">刷新申诉进度</button>
    </div>
    <p class="muted">
      可以对自己主题的隐藏、锁定或账号禁言提出申诉。禁言期间仍可提交。处理结果和说明保留在这里。
    </p>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    <article v-for="a in actions" :key="a.id" class="panel">
      <strong>{{ names[a.action] }} · {{ a.title || "当前账号" }}</strong>
      <p>管理原因：{{ a.reason }}</p>
      <p v-if="a.expires_at">
        截止 {{ new Date(a.expires_at).toLocaleString() }}
      </p>
      <button
        v-if="!a.appeal_id"
        @click="
          selected = a;
          reason = '';
        "
      >
        提交申诉</button
      ><span v-else class="badge">已提交</span>
    </article>
    <form v-if="selected" class="panel" @submit.prevent="submit">
      <h3>申诉：{{ selected.title || "账号禁言" }}</h3>
      <label class="field"
        >申诉理由<textarea
          v-model="reason"
          minlength="3"
          maxlength="1000"
          required
          :disabled="busy"
        />
      </label>
      <div class="actions">
        <button class="primary" :disabled="busy || reason.trim().length < 3">
          发送申诉</button
        ><button type="button" :disabled="busy" @click="selected = null">
          取消
        </button>
      </div>
    </form>
    <p v-if="!actions.length && !appeals.length">暂无需要处理的管理记录。</p>
    <article v-for="a in appeals" :key="a.id" class="panel">
      <span class="badge">{{ states[a.status] }}</span>
      {{ a.title || names[a.action] }}
      <p>申诉理由：{{ a.reason }}</p>
      <p v-if="a.decision">处理说明：{{ a.decision }}</p>
      <p class="muted">
        {{ new Date(a.created_at).toLocaleString() }} · #{{ a.id }}
      </p>
    </article>
    <button
      v-if="appeals.length && appeals.length % 50 === 0"
      @click="load(true)"
    >
      更早申诉
    </button>
  </section>
</template>
