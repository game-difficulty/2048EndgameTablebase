<script setup>
import { inject, ref, watch } from "vue";
import { api } from "../api";
const { session, user, boards } = inject("forum");
const tab = ref("announcements"),
  items = ref([]),
  error = ref(""),
  status = ref(""),
  busy = ref(false),
  editing = ref(null),
  title = ref(""),
  text = ref(""),
  board = ref("announcements"),
  sites = ref(["forum"]),
  publish = ref(""),
  expires = ref(""),
  allowReplies = ref(true),
  notifyAll = ref(false),
  schedule = ref(false),
  revision = ref(0),
  reason = ref(""),
  source = ref("competition"),
  sourceId = ref(""),
  sourceRevision = ref(1),
  sourcePath = ref(""),
  sourceStatus = ref("published"),
  topicId = ref("");
let epoch = 0;
const labels = {
  draft: "草稿",
  scheduled: "已排期",
  published: "已发布",
  expired: "已到期",
  retracted: "已撤回",
  open: "待处理",
  completed: "已完成",
  rejected: "已结案",
};
const metrics = ref(null);
async function getMetrics() {
  try {
    metrics.value = await api("/moderation/metrics");
  } catch (e) {
    error.value = e.message;
  }
}
function localDate(value) {
  if (!value) return "";
  const d = new Date(value);
  return new Date(d.getTime() - d.getTimezoneOffset() * 60000)
    .toISOString()
    .slice(0, 16);
}
async function load() {
  const ticket = ++epoch;
  items.value = [];
  if (!session.value?.is_admin) return;
  try {
    const r = await api("/moderation/" + tab.value);
    if (ticket === epoch) items.value = r.items;
  } catch (e) {
    if (ticket === epoch) error.value = e.message;
  }
}
function edit(item = null) {
  editing.value = item;
  title.value = item?.payload?.title || item?.title || "";
  text.value =
    item?.payload?.body?.blocks
      ?.filter((b) => b.type === "paragraph")
      .map((b) => b.text)
      .join("\n\n") ||
    item?.summary ||
    "";
  board.value = item?.payload?.board_slug || "announcements";
  sites.value = item?.sites || ["forum"];
  publish.value = localDate(item?.publish_at);
  expires.value = localDate(item?.expires_at);
  allowReplies.value = item?.allow_replies ?? true;
  notifyAll.value = item?.notify_all || false;
  schedule.value = item?.status === "scheduled";
  revision.value = item?.revision || 0;
  reason.value = "";
  source.value = item?.source || "competition";
  sourceId.value = item?.source_id || "";
  sourceRevision.value = (item?.revision || 0) + 1;
  sourcePath.value = item?.path || "";
  sourceStatus.value = item?.status === "withdrawn" ? "withdrawn" : "published";
  topicId.value = item?.topic_id || "";
}
async function run(fn) {
  busy.value = true;
  error.value = "";
  status.value = "";
  try {
    await fn();
    edit();
    await load();
    status.value = "操作已保存并记录审计";
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
function saveAnnouncement() {
  run(() =>
    api(
      "/moderation/announcements" +
        (editing.value ? "/" + editing.value.id : ""),
      {
        method: editing.value ? "PUT" : "POST",
        body: {
          topic: {
            board_slug: board.value,
            title: title.value,
            body: {
              version: 1,
              blocks: [{ type: "paragraph", text: text.value }],
            },
            tags: [],
            kind: "discussion",
            poll: null,
          },
          sites: sites.value,
          publish_at: publish.value
            ? new Date(publish.value).toISOString()
            : null,
          expires_at: expires.value
            ? new Date(expires.value).toISOString()
            : null,
          allow_replies: allowReplies.value,
          notify_all: notifyAll.value,
          schedule: schedule.value,
          revision: revision.value,
        },
      },
    ),
  );
}
function saveCard() {
  run(() =>
    api("/moderation/external-cards", {
      method: "PUT",
      body: {
        source: source.value,
        source_id: sourceId.value,
        revision: Number(sourceRevision.value),
        title: title.value,
        summary: text.value,
        path: sourcePath.value,
        status: sourceStatus.value,
        topic_id: Number(topicId.value) || null,
        reason: reason.value,
      },
    }),
  );
}
watch(
  [tab, () => user.value?.id, () => session.value?.is_admin],
  () => {
    edit();
    load();
  },
  { immediate: true },
);
</script>
<template>
  <header class="heading">
    <div>
      <p class="eyebrow">管理工作台 / 社区运营</p>
      <h1>公告与运营</h1>
      <p class="muted">安排社区公告，维护来源动态，处理隐私请求。</p>
    </div>
    <RouterLink class="button" to="/moderation">内容与审核 →</RouterLink>
  </header>
  <p v-if="!session?.is_admin" class="empty">需要管理员权限。</p>
  <template v-else>
    <nav class="view-tabs" aria-label="运营功能">
      <button
        :aria-pressed="tab === 'announcements'"
        @click="tab = 'announcements'"
      >
        定时公告</button
      ><button
        :aria-pressed="tab === 'external-cards'"
        @click="tab = 'external-cards'"
      >
        赛事与直播动态</button
      ><button :aria-pressed="tab === 'privacy'" @click="tab = 'privacy'">
        隐私请求</button
      ><button @click="load">刷新</button>
    </nav>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    <p role="status">{{ status }}</p>
    <details class="panel">
      <summary @click="getMetrics">运行状态与队列指标</summary>
      <template v-if="metrics"
        ><p>
          主题 {{ metrics.content.topics }} · 隐藏
          {{ metrics.content.hidden }} · 附件 {{ metrics.media.files }}（{{
            (metrics.media.bytes / 1048576).toFixed(1)
          }}
          MiB）
        </p>
        <p>
          待投递 {{ metrics.queue.pending }} · 重试
          {{ metrics.queue.retrying }} · 最早等待
          {{ Math.round(metrics.queue.oldest_seconds) }} 秒
        </p>
        <p>
          待处理举报 {{ metrics.open_reports }} · 申诉
          {{ metrics.open_appeals }}
        </p>
        <button @click="getMetrics">刷新指标</button></template
      >
    </details>
    <div
      class="operations-layout"
      :class="{ 'records-only': tab === 'privacy' }"
    >
      <div v-if="tab !== 'privacy'" class="operations-editor">
        <form
          v-if="tab === 'announcements'"
          class="panel"
          @submit.prevent="saveAnnouncement"
        >
          <fieldset class="editor-fieldset" :disabled="busy">
            <h2>{{ editing ? "编辑公告 #" + editing.id : "新建公告" }}</h2>
            <label class="field"
              >公告标题<input
                v-model="title"
                minlength="3"
                maxlength="120"
                required /></label
            ><label class="field"
              >公告正文<textarea
                v-model="text"
                rows="6"
                maxlength="20000"
                required
              /></label
            ><label class="field"
              >发布板块<select v-model="board">
                <option v-for="b in boards" :key="b.slug" :value="b.slug">
                  {{ b.name }}
                </option>
              </select></label
            >
            <div class="actions">
              <label
                v-for="(label, key) in {
                  forum: '论坛',
                  main: '主站',
                  play: 'Play',
                  competition: '赛事',
                  live: '直播',
                }"
                :key="key"
                ><input v-model="sites" type="checkbox" :value="key" />{{
                  label
                }}</label
              >
            </div>
            <div class="filter-grid">
              <label class="field"
                >发布时间<input
                  v-model="publish"
                  type="datetime-local"
                  :required="schedule" /></label
              ><label class="field"
                >到期时间（可选）<input v-model="expires" type="datetime-local"
              /></label>
            </div>
            <label class="poll-option"
              ><input v-model="allowReplies" type="checkbox" />允许回复</label
            ><label class="poll-option"
              ><input
                v-model="notifyAll"
                type="checkbox"
              />通知已加入社区的所有用户（遵守通知偏好）</label
            ><label class="poll-option"
              ><input v-model="schedule" type="checkbox" />按上述时间发布</label
            >
            <p class="muted">
              未勾选排期时保存为草稿。跨站公告摘要由公开接口提供，目标站点需接入后才会显示。
            </p>
            <div class="actions">
              <button class="primary" :disabled="!sites.length">
                {{ schedule ? "保存排期" : "保存公告草稿" }}</button
              ><button v-if="editing" type="button" @click="edit()">
                取消编辑
              </button>
            </div>
          </fieldset>
        </form>
        <form
          v-if="tab === 'external-cards'"
          class="panel"
          @submit.prevent="saveCard"
        >
          <fieldset class="editor-fieldset" :disabled="busy">
            <h2>发布或修订来源动态</h2>
            <p class="muted">
              由管理员核对来源后登记。这里只展示动态和来源链接，不认证录像或成绩。撤回后从公开列表移除。
            </p>
            <label class="field"
              >来源<select v-model="source">
                <option value="competition">赛事站</option>
                <option value="live">直播站</option>
              </select></label
            ><label class="field"
              >来源记录 ID<input
                v-model="sourceId"
                maxlength="80"
                pattern="[a-zA-Z0-9_-]+"
                required /></label
            ><label class="field"
              >来源版本<input
                v-model="sourceRevision"
                type="number"
                min="1"
                required /></label
            ><label class="field"
              >标题<input
                v-model="title"
                minlength="3"
                maxlength="120"
                required /></label
            ><label class="field"
              >摘要<textarea v-model="text" maxlength="2000" /></label
            ><label class="field"
              >来源站内路径<input
                v-model="sourcePath"
                placeholder="/events/记录ID"
                maxlength="300"
                required /></label
            ><label class="field"
              >关联讨论 ID（可选）<input
                v-model="topicId"
                type="number"
                min="1" /></label
            ><label class="field"
              >状态<select v-model="sourceStatus">
                <option value="published">公开</option>
                <option value="withdrawn">来源已撤回</option>
              </select></label
            ><label class="field"
              >核对与修订说明<textarea
                v-model="reason"
                minlength="3"
                maxlength="1000"
                required
              /></label
            ><button class="primary">保存来源动态</button>
          </fieldset>
        </form>
      </div>
      <section class="operations-records">
        <h2>
          {{
            tab === "announcements"
              ? "公告记录"
              : tab === "privacy"
                ? "待核查与已结案请求"
                : "来源动态记录"
          }}
        </h2>
        <p v-if="!items.length" class="empty">暂无记录。</p>
        <article v-for="item in items" :key="item.id" class="panel">
          <template v-if="tab === 'announcements'"
            ><h2>{{ item.payload.title }}</h2>
            <p>
              <span class="badge">{{ labels[item.status] }}</span> · 版本
              {{ item.revision }} · #{{ item.id }}
            </p>
            <p v-if="item.publish_at">
              发布时间：{{ new Date(item.publish_at).toLocaleString() }}
            </p>
            <RouterLink v-if="item.topic_id" :to="'/t/' + item.topic_id"
              >查看并修订公告正文</RouterLink
            >
            <div class="actions">
              <button
                v-if="['draft', 'scheduled'].includes(item.status)"
                @click="edit(item)"
              >
                编辑排期</button
              ><button
                v-if="item.status !== 'retracted'"
                :disabled="busy"
                @click="
                  run(() =>
                    api(
                      `/moderation/announcements/${item.id}?revision=${item.revision}`,
                      {
                        method: 'DELETE',
                        body: { reason: '管理员主动撤回公告' },
                      },
                    ),
                  )
                "
              >
                撤回公告
              </button>
            </div></template
          >
          <template v-else-if="tab === 'external-cards'"
            ><h2>{{ item.title }}</h2>
            <p>
              {{ item.source }} · {{ item.source_id }} · 版本
              {{ item.revision }} ·
              {{ item.status === "withdrawn" ? "已撤回" : "公开" }}
            </p>
            <button @click="edit(item)">登记下一版本</button></template
          >
          <template v-else
            ><h2>隐私请求 #{{ item.id }} · 用户 #{{ item.user_id }}</h2>
            <p>{{ item.reason }}</p>
            <p>{{ labels[item.status] }} {{ item.decision }}</p>
            <template v-if="item.status === 'open'"
              ><label class="field"
                >实际处理范围与说明<textarea
                  v-model="reason"
                  maxlength="1000"
                />
              </label>
              <p class="muted">
                完成标记仅记录处理结果；请先核对并完成用户请求，不会自动删除账号或批量抹除内容。
              </p>
              <div class="actions">
                <button
                  :disabled="busy || reason.trim().length < 3"
                  @click="
                    run(() =>
                      api('/moderation/privacy/' + item.id, {
                        method: 'POST',
                        body: { decision: 'completed', reason },
                      }),
                    )
                  "
                >
                  记录为已完成</button
                ><button
                  :disabled="busy || reason.trim().length < 3"
                  @click="
                    run(() =>
                      api('/moderation/privacy/' + item.id, {
                        method: 'POST',
                        body: { decision: 'rejected', reason },
                      }),
                    )
                  "
                >
                  说明并结案
                </button>
              </div></template
            ></template
          >
        </article>
      </section>
    </div>
  </template>
</template>
