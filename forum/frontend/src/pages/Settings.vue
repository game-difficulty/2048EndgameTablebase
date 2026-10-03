<script setup>
import { inject, ref, watch } from "vue";
import { api } from "../api";
const { user, boards } = inject("forum"),
  prefs = ref({ enabled: true, categories: {} }),
  blocks = ref([]),
  requests = ref([]),
  error = ref(""),
  status = ref(""),
  busy = ref(false),
  ready = ref(false),
  blockKind = ref("user"),
  target = ref(""),
  reason = ref("");
const names = {
  reply: "回复我的内容",
  mention: "提及我",
  subscription: "主题与板块订阅",
  follow: "关注作者的新主题",
  moderation: "管理与申诉结果",
  system: "系统与全站公告",
};
let epoch = 0;
async function load() {
  const t = ++epoch;
  ready.value = false;
  blocks.value = [];
  requests.value = [];
  prefs.value = { enabled: true, categories: {} };
  if (!user.value) return;
  try {
    const [p, b, r] = await Promise.all([
      api("/notification-settings"),
      api("/blocks"),
      api("/privacy/requests"),
    ]);
    if (t !== epoch) return;
    prefs.value = p;
    blocks.value = b.items;
    requests.value = r.items;
    ready.value = true;
  } catch (e) {
    if (t === epoch) error.value = e.message;
  }
}
async function run(fn) {
  if (busy.value || !ready.value) return;
  busy.value = true;
  error.value = "";
  status.value = "";
  try {
    await fn();
    await load();
    status.value = "已保存";
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
async function exportData() {
  await run(async () => {
    const data = await api("/privacy/export");
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "forum-data.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}
watch(() => user.value?.id, load, { immediate: true });
</script>
<template>
  <header class="heading">
    <div>
      <p class="eyebrow">个人空间 / 社区设置</p>
      <h1>偏好与隐私</h1>
      <p class="muted">选择想接收的消息，管理你的社区数据。</p>
    </div>
    <RouterLink class="button" to="/community">返回我的社区</RouterLink>
  </header>
  <p v-if="!user" class="empty">请先登录。</p>
  <template v-else>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    <p role="status">{{ status }}</p>
    <p v-if="!ready && !error" role="status">正在读取社区设置…</p>
    <button v-if="!ready && error" @click="load">重新读取设置</button>
    <fieldset class="editor-fieldset settings-grid" :disabled="busy || !ready">
      <section class="panel">
        <h2>通知偏好</h2>
        <label class="poll-option"
          ><input type="checkbox" v-model="prefs.enabled" />接收新通知</label
        ><label v-for="(label, key) in names" :key="key" class="poll-option"
          ><input
            type="checkbox"
            :checked="prefs.categories[key] !== false"
            @change="prefs.categories[key] = $event.target.checked"
          />{{ label }}</label
        ><button
          :disabled="busy"
          @click="
            run(() =>
              api('/notification-settings', { method: 'PUT', body: prefs }),
            )
          "
        >
          保存通知偏好
        </button>
      </section>
      <section class="panel">
        <h2>屏蔽用户与板块</h2>
        <p class="muted">
          屏蔽后不再显示相应信息流和通知，用户回复保留楼层占位。公开内容仍可通过其他账号浏览。
        </p>
        <label class="field"
          >屏蔽类型<select v-model="blockKind">
            <option value="user">用户</option>
            <option value="board">板块</option>
          </select></label
        ><label v-if="blockKind === 'user'" class="field"
          >用户 ID<input v-model="target" type="number" min="1" /></label
        ><label v-else class="field"
          >板块<select v-model="target">
            <option v-for="b in boards" :value="b.id" :key="b.id">
              {{ b.name }}
            </option>
          </select></label
        ><button
          :disabled="busy || !target"
          @click="
            run(() => api(`/blocks/${blockKind}/${target}`, { method: 'PUT' }))
          "
        >
          添加屏蔽
        </button>
        <article
          v-for="b in blocks"
          :key="b.kind + b.target_id"
          class="actions"
        >
          <span>{{ b.name }} · {{ b.kind === "user" ? "用户" : "板块" }}</span
          ><button
            :disabled="busy"
            @click="
              run(() =>
                api(`/blocks/${b.kind}/${b.target_id}`, { method: 'DELETE' }),
              )
            "
          >
            解除屏蔽
          </button>
        </article>
      </section>
      <section class="panel">
        <h2>数据与隐私</h2>
        <p>
          导出自己的帖子、草稿、收藏与设置。每类最多 1000
          条，超出时导出文件会注明；可以提交完整导出或删除请求，由管理员核对范围后处理。
        </p>
        <button :disabled="busy" @click="exportData">下载我的社区数据</button
        ><label class="field"
          >完整导出 / 删除 / 隐私处理请求<textarea
            v-model="reason"
            maxlength="1000"
          /></label
        ><button
          :disabled="busy || reason.trim().length < 3"
          @click="
            run(() =>
              api('/privacy/requests', { method: 'POST', body: { reason } }),
            )
          "
        >
          提交隐私请求
        </button>
        <article v-for="r in requests" :key="r.id">
          <p>
            <span class="badge">{{
              { open: "待处理", completed: "已完成", rejected: "已结案" }[
                r.status
              ]
            }}</span>
            {{ r.reason }}
          </p>
          <p v-if="r.decision">{{ r.decision }}</p>
        </article>
      </section>
    </fieldset>
  </template>
</template>
