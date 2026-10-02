<script setup>
import { computed, inject, ref, watch } from "vue";
import { api } from "../api";
import Document from "../components/Document.vue";
const { user, session, boards, refresh } = inject("forum");
const section = ref("reports"),
  items = ref([]),
  error = ref(""),
  loading = ref(false),
  busy = ref(false),
  query = ref(""),
  reason = ref(""),
  selected = ref(null),
  action = ref(""),
  boardId = ref(""),
  hours = ref(24),
  editBoard = ref(null);
const sections = computed(() => [
  ["reports", "举报处理"],
  ["appeals", "申诉处理"],
  ["topics", "主题管理"],
  ["audit", "操作审计"],
  ...(session.value?.is_admin
    ? [
        ["users", "用户权限"],
        ["boards", "板块设置"],
        ["media", "附件管理"],
        ["jobs", "通知任务"],
      ]
    : []),
]);
const actionNames = {
  accepted: "通过申诉并撤销该处置",
  rejected: "说明原因并结案",
  resolve: "举报结案",
  hide: "隐藏主题",
  restore: "恢复公开",
  lock: "锁定主题",
  unlock: "解锁主题",
  pin: "置顶",
  unpin: "取消置顶",
  grant: "授予版主",
  revoke: "撤销版主",
  mute: "禁言",
  unmute: "解除禁言",
  remove_media: "移除附件",
  edit_board: "编辑板块",
};
let epoch = 0;
async function load(more = false) {
  const ticket = ++epoch;
  error.value = "";
  if (!more) items.value = [];
  if (!session.value?.can_moderate) return;
  if (section.value === "boards") return;
  loading.value = true;
  try {
    const params = new URLSearchParams({ q: query.value });
    if (more && items.value.length)
      params.set("before", items.value.at(-1).id || items.value.at(-1).user_id);
    const r = await api(
      section.value === "appeals"
        ? `/moderation/appeals?${params}`
        : `/moderation/overview/${section.value}?${params}`,
    );
    if (ticket === epoch)
      items.value = more ? [...items.value, ...r.items] : r.items;
  } catch (e) {
    if (ticket === epoch) error.value = e.message;
  } finally {
    if (ticket === epoch) loading.value = false;
  }
}
function choose(item, value) {
  selected.value = item;
  action.value = value;
  reason.value = "";
  boardId.value = boards.value[0]?.id || "";
  hours.value = 24;
  editBoard.value = value === "edit_board" ? { ...item } : null;
}
async function submit() {
  if (!selected.value || reason.value.trim().length < 3) return;
  busy.value = true;
  error.value = "";
  try {
    const item = selected.value,
      r = reason.value.trim();
    if (["accepted", "rejected"].includes(action.value))
      await api("/moderation/appeals/" + item.id, {
        method: "POST",
        body: { decision: action.value, reason: r },
      });
    else if (action.value === "resolve")
      await api(`/moderation/reports/${item.id}/resolve`, {
        method: "POST",
        body: { reason: r },
      });
    else if (
      ["hide", "restore", "lock", "unlock", "pin", "unpin"].includes(
        action.value,
      )
    )
      await api(`/topics/${item.topic_id || item.id}/moderation`, {
        method: "POST",
        body: { action: action.value, reason: r },
      });
    else if (["grant", "revoke", "mute", "unmute"].includes(action.value))
      await api(`/moderation/users/${item.user_id}`, {
        method: "POST",
        body: {
          action: action.value,
          reason: r,
          board_id: Number(boardId.value) || null,
          hours: Number(hours.value),
        },
      });
    else if (action.value === "remove_media")
      await api("/media/" + item.id, { method: "DELETE", body: { reason: r } });
    else if (action.value === "edit_board") {
      const b = editBoard.value;
      await api("/moderation/boards/" + item.id, {
        method: "PATCH",
        body: {
          name: b.name,
          description: b.description,
          position: Number(b.position),
          staff_only: b.staff_only,
          reason: r,
        },
      });
      await refresh();
    }
    selected.value = null;
    await load();
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
watch(
  [
    () => section.value,
    () => user.value?.id,
    () => session.value?.can_moderate,
  ],
  () => {
    selected.value = null;
    query.value = "";
    load();
  },
  { immediate: true },
);
</script>
<template>
  <div class="heading">
    <div>
      <p class="eyebrow">COMMUNITY ADMINISTRATION</p>
      <h1>社区管理</h1>
    </div>
    <button :disabled="loading" @click="load()">刷新</button>
  </div>
  <p v-if="!session?.can_moderate" class="empty">需要相应板块的管理权限。</p>
  <template v-else>
    <nav class="actions" aria-label="管理功能">
      <button
        v-for="[key, label] in sections"
        :key="key"
        :aria-pressed="section === key"
        @click="section = key"
      >
        {{ label }}
      </button>
    </nav>
    <form
      v-if="['topics', 'users'].includes(section)"
      class="search"
      @submit.prevent="load()"
    >
      <input
        v-model="query"
        :aria-label="section === 'users' ? '搜索用户昵称或 ID' : '搜索主题'"
        placeholder="搜索"
        maxlength="80"
      /><button>查询</button>
    </form>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    <p v-if="loading" role="status">正在加载…</p>
    <form
      v-if="selected"
      class="panel moderation-form"
      @submit.prevent="submit"
    >
      <h2>{{ actionNames[action] }}</h2>
      <p>
        对象：{{
          selected.title ||
          selected.display_name ||
          selected.name ||
          selected.id
        }}
      </p>
      <label v-if="['grant', 'revoke'].includes(action)" class="field"
        >指定板块<select v-model="boardId">
          <option v-for="b in boards" :key="b.id" :value="b.id">
            {{ b.name }}
          </option>
        </select></label
      >
      <label v-if="action === 'mute'" class="field"
        >禁言小时数<input
          v-model="hours"
          type="number"
          min="1"
          max="8760"
          required
      /></label>
      <template v-if="editBoard"
        ><label class="field"
          >板块名称<input
            v-model="editBoard.name"
            maxlength="80"
            required /></label
        ><label class="field"
          >板块说明<textarea
            v-model="editBoard.description"
            maxlength="1000"
          /></label
        ><label class="field"
          >排序<input
            v-model="editBoard.position"
            type="number"
            min="0"
            max="10000" /></label
        ><label
          ><input
            v-model="editBoard.staff_only"
            type="checkbox"
          />仅管理人员可发主题</label
        ></template
      >
      <label class="field"
        >处理原因<textarea
          v-model="reason"
          minlength="3"
          maxlength="1000"
          required
        />
      </label>
      <p v-if="action === 'remove_media'" class="notice">
        移除后将释放附件数据，所有引用位置都无法继续显示；此操作不能撤销。
      </p>
      <div class="actions">
        <button class="primary" :disabled="busy || reason.trim().length < 3">
          执行并记录审计</button
        ><button type="button" :disabled="busy" @click="selected = null">
          取消
        </button>
      </div>
    </form>
    <template v-if="section === 'boards'"
      ><article v-for="b in boards" :key="b.id" class="panel">
        <h2>{{ b.name }}</h2>
        <p>{{ b.description }}</p>
        <p class="muted">
          {{ b.slug }} · 排序 {{ b.position }} ·
          {{ b.staff_only ? "仅管理人员发帖" : "开放发帖" }}
        </p>
        <button @click="choose(b, 'edit_board')">编辑板块</button>
      </article></template
    >
    <p v-else-if="!loading && !items.length" class="empty">暂无记录。</p>
    <article v-for="item in items" :key="item.id || item.user_id" class="panel">
      <template v-if="section === 'appeals'"
        ><h2>{{ item.title || "账号禁言申诉" }}</h2>
        <p>
          申请人：{{ item.display_name }} · #{{ item.user_id }} ·
          {{
            { open: "等待处理", accepted: "申诉通过", rejected: "已结案" }[
              item.status
            ] || item.status
          }}
        </p>
        <p>
          原管理原因：{{ item.moderation_reason }}（操作人 #{{
            item.operator_id
          }}）
        </p>
        <p>申诉理由：{{ item.reason }}</p>
        <p v-if="item.decision">处理说明：{{ item.decision }}</p>
        <div v-if="item.status === 'open'" class="actions">
          <button @click="choose(item, 'accepted')">通过申诉</button
          ><button @click="choose(item, 'rejected')">说明并结案</button>
        </div></template
      >
      <template v-else-if="section === 'reports'"
        ><RouterLink :to="`/t/${item.topic_id}#p-${item.post_id}`"
          >{{ item.title }} · #{{ item.post_number }}</RouterLink
        >
        <p><strong>举报原因：</strong>{{ item.reason }}</p>
        <details>
          <summary>查看被举报正文</summary>
          <Document :body="item.body" />
        </details>
        <div class="actions">
          <button @click="choose(item, 'resolve')">结案并说明</button
          ><button @click="choose(item, 'hide')">隐藏主题</button
          ><button @click="choose(item, 'lock')">锁定主题</button>
        </div></template
      >
      <template v-else-if="section === 'topics'"
        ><RouterLink :to="'/t/' + item.id">{{ item.title }}</RouterLink>
        <p class="muted">
          {{ item.status === "published" ? "公开" : "已隐藏" }} ·
          {{ item.locked ? "已锁定" : "可回复" }} ·
          {{ item.pinned ? "置顶" : "普通" }} · 作者 #{{ item.author_id }}
        </p>
        <div class="actions">
          <button
            @click="
              choose(item, item.status === 'published' ? 'hide' : 'restore')
            "
          >
            {{ item.status === "published" ? "隐藏" : "恢复公开" }}</button
          ><button @click="choose(item, item.locked ? 'unlock' : 'lock')">
            {{ item.locked ? "解锁" : "锁定" }}</button
          ><button @click="choose(item, item.pinned ? 'unpin' : 'pin')">
            {{ item.pinned ? "取消置顶" : "置顶" }}
          </button>
        </div></template
      >
      <template v-else-if="section === 'users'"
        ><h2>
          {{ item.display_name }} <small>#{{ item.user_id }}</small>
        </h2>
        <p>
          版主板块：{{
            (item.boards || [])
              .map((id) => boards.find((b) => b.id === id)?.name || id)
              .join("、") || "无"
          }}
        </p>
        <p v-if="item.expires_at">
          禁言至 {{ new Date(item.expires_at).toLocaleString() }} ·
          {{ item.sanction_reason }}
        </p>
        <div class="actions">
          <button @click="choose(item, 'grant')">授予版主</button
          ><button @click="choose(item, 'revoke')">撤销版主</button
          ><button @click="choose(item, 'mute')">禁言</button
          ><button v-if="item.expires_at" @click="choose(item, 'unmute')">
            解除禁言
          </button>
        </div></template
      >
      <template v-else-if="section === 'audit'"
        ><strong
          >{{ item.action }} · {{ item.target_type }} #{{
            item.target_id || item.topic_id
          }}</strong
        >
        <p>{{ item.reason }}</p>
        <p class="muted">
          操作人 #{{ item.operator_id }} ·
          {{ new Date(item.created_at).toLocaleString() }}
        </p>
        <details>
          <summary>变更前记录</summary>
          <pre>{{ JSON.stringify(item.previous, null, 2) }}</pre>
        </details></template
      >
      <template v-else-if="section === 'media'"
        ><strong>{{ item.kind }} · {{ item.status }}</strong>
        <p class="muted">
          {{ item.id }} · 作者 #{{ item.owner_id }} ·
          {{ (item.size / 1024).toFixed(1) }} KiB
        </p>
        <button
          v-if="item.status === 'active'"
          @click="choose(item, 'remove_media')"
        >
          移除附件
        </button></template
      >
      <template v-else-if="section === 'jobs'"
        ><strong>#{{ item.id }} · {{ item.kind }}</strong>
        <p>
          {{ item.delivered_at ? "已处理" : "待处理" }} · 重试
          {{ item.attempts }} 次
        </p>
        <p v-if="item.last_error" class="notice error">
          {{ item.last_error }}
        </p></template
      >
    </article>
    <button
      v-if="items.length >= 50 && !['media', 'boards'].includes(section)"
      :disabled="loading"
      @click="load(true)"
    >
      加载更早记录
    </button>
  </template>
</template>
