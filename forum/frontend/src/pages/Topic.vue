<script setup>
import { inject, nextTick, ref, watch } from "vue";
import {
  useRoute,
  useRouter,
  onBeforeRouteLeave,
  onBeforeRouteUpdate,
} from "vue-router";
import { api, submission } from "../api";
import Document from "../components/Document.vue";
import BoardSyntaxHelp from "../components/BoardSyntaxHelp.vue";
import BoardSyntaxPreview from "../components/BoardSyntaxPreview.vue";
import { insertAtCursor } from "../editorInsertion";
import RichTools from "../components/RichTools.vue";
import SubscribeButton from "../components/SubscribeButton.vue";
import TopicExtensions from "../components/TopicExtensions.vue";
import { usePasteImage } from "../pasteImage";
import { useReplyDraft } from "../replyDraft";
import { useReadingPosition } from "../readingPosition";
const replyInput = ref(null);
const authorOnly = ref(false),
  revisions = ref(null);
async function history(post) {
  const ticket = generation;
  await action(async () => {
    const result = await api("/posts/" + post.id + "/revisions");
    if (ticket === generation) revisions.value = result.items;
  });
}
function insertSyntax(snippet, editing = false) {
  insertAtCursor(
    editing ? editText : reply,
    editing
      ? document.getElementById(`edit-body-${editId.value}`)
      : replyInput.value,
    snippet,
  );
}
const route = useRoute(),
  router = useRouter(),
  { user } = inject("forum");
const data = ref(null),
  error = ref(""),
  loading = ref(false),
  busy = ref(false),
  reply = ref(""),
  replyTo = ref(null),
  editId = ref(null),
  editText = ref("");
const reportPost = ref(null),
  reason = ref(""),
  modAction = ref(""),
  modReason = ref(""),
  status = ref("");
const key = submission();
const pasteImage = usePasteImage(
  () => `${user.value?.id}:${route.params.id}:${editId.value}`,
);
function paste(event, editing = false) {
  pasteImage.paste(event, editing ? editText : reply);
}
const draft = useReplyDraft(
  () => route.params.id,
  () => user.value?.id,
  reply,
  replyTo,
);
const reading = useReadingPosition(
  () => route.params.id,
  () => user.value?.id,
  data,
);
async function leave() {
  await Promise.all([draft.save(), reading.flush()]);
}
onBeforeRouteLeave(leave);
onBeforeRouteUpdate(async (to, from) => {
  if (to.params.id !== from.params.id) await leave();
});
const metadataOpen = ref(false),
  topicTitle = ref(""),
  topicTags = ref("");
function openMetadata() {
  topicTitle.value = data.value.topic.title;
  topicTags.value = data.value.topic.tags.join("、");
  metadataOpen.value = true;
}
async function saveMetadata() {
  await action(async () => {
    await api("/topics/" + data.value.topic.id, {
      method: "PATCH",
      body: {
        title: topicTitle.value,
        tags: topicTags.value
          .split(/[、,，]/)
          .map((t) => t.trim())
          .filter(Boolean),
        revision: data.value.topic.revision,
      },
    });
    metadataOpen.value = false;
    await load();
  });
}
let generation = 0;
async function load(more = false, fromStart = false) {
  const ticket = ++generation;
  loading.value = true;
  error.value = "";
  try {
    const focus = /^#p-([1-9][0-9]*)$/.exec(route.hash)?.[1];
    const query = more
      ? `?after=${data.value.next_after}`
      : !fromStart && focus
        ? `?focus_post=${focus}`
        : "";
    const result = await api(
      `/topics/${route.params.id}${query}${query ? "&" : "?"}author_only=${authorOnly.value}`,
    );
    if (ticket !== generation) return;
    data.value = more
      ? { ...result, posts: [...data.value.posts, ...result.posts] }
      : result;
    await nextTick();
    if (!more && !fromStart && route.hash)
      document.getElementById(route.hash.slice(1))?.scrollIntoView();
  } catch (e) {
    if (ticket === generation) error.value = e.message;
  } finally {
    if (ticket === generation) loading.value = false;
  }
}
async function action(fn) {
  if (busy.value) return;
  busy.value = true;
  error.value = "";
  status.value = "";
  try {
    await fn();
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
async function send() {
  await action(async () => {
    const body = {
      body: { version: 1, blocks: [{ type: "paragraph", text: reply.value }] },
      reply_to: replyTo.value?.id || null,
    };
    const result = await api(`/topics/${route.params.id}/posts`, {
      method: "POST",
      body,
      key: key(body),
    });
    key.reset();
    await draft.published();
    status.value = "回复已发布。";
    const hash = `#p-${result.post_id}`;
    if (route.hash === hash) await load();
    else await router.replace({ hash });
  });
}
async function like(post) {
  await action(async () => {
    await api(`/posts/${post.id}/like`, {
      method: post.liked ? "DELETE" : "PUT",
    });
    post.likes += post.liked ? -1 : 1;
    post.liked = !post.liked;
  });
}
async function bookmark() {
  await action(async () => {
    await api(`/topics/${route.params.id}/bookmark`, {
      method: data.value.topic.bookmarked ? "DELETE" : "PUT",
    });
    data.value.topic.bookmarked = !data.value.topic.bookmarked;
  });
}
function edit(post) {
  editId.value = post.id;
  editText.value = post.body.blocks
    .filter((b) => b.type === "paragraph")
    .map((b) => b.text)
    .join("\n\n");
}
async function save(post) {
  await action(async () => {
    const blocks = [...post.body.blocks.filter((b) => b.type !== "paragraph")];
    if (editText.value.trim())
      blocks.unshift({ type: "paragraph", text: editText.value });
    await api(`/posts/${post.id}`, {
      method: "PATCH",
      body: { revision: post.revision, body: { version: 1, blocks } },
    });
    editId.value = null;
    await load();
  });
}
async function remove(post) {
  if (!confirm("删除这条回复？楼层位置会保留。")) return;
  await action(async () => {
    await api(`/posts/${post.id}?revision=${post.revision}`, {
      method: "DELETE",
    });
    await load();
  });
}
async function report() {
  await action(async () => {
    await api(`/posts/${reportPost.value}/reports`, {
      method: "POST",
      body: { reason: reason.value },
    });
    reportPost.value = null;
    reason.value = "";
    status.value = "举报已提交，管理人员会处理。";
  });
}
async function moderate() {
  await action(async () => {
    await api(`/topics/${route.params.id}/moderation`, {
      method: "POST",
      body: { action: modAction.value, reason: modReason.value },
    });
    modAction.value = "";
    modReason.value = "";
    await load();
  });
}
watch(
  [() => route.params.id, () => user.value?.id],
  () => {
    data.value = null;
    revisions.value = null;
    editId.value = null;
    reportPost.value = null;
    metadataOpen.value = false;
    load();
  },
  { immediate: true },
);
watch(
  () => route.hash,
  () => load(),
);
watch(authorOnly, () => load());
</script>
<template>
  <RouterLink to="/" class="back">← 返回讨论</RouterLink>
  <div v-if="error" class="notice error" role="alert">
    {{ error }} <button @click="load()">重新加载</button>
  </div>
  <p v-if="loading && !data" class="empty">正在加载主题…</p>
  <template v-if="data"
    ><div class="heading">
      <div>
        <RouterLink :to="'/c/' + data.topic.board_slug" class="eyebrow">{{
          data.topic.board_name
        }}</RouterLink>
        <h1>{{ data.topic.title }}</h1>
        <div class="meta">
          <span v-for="tag in data.topic.tags" :key="tag">#{{ tag }}</span
          ><span v-if="data.topic.locked">已锁定</span
          ><span v-if="data.topic.status !== 'published'"
            >已隐藏，仅作者和管理人员可见</span
          >
        </div>
      </div>
      <button v-if="user" :disabled="busy" @click="bookmark">
        {{ data.topic.bookmarked ? "已收藏" : "收藏主题" }}
      </button>
      <SubscribeButton kind="topic" :id="data.topic.id" />
      <button
        v-if="
          user &&
          (user.id === data.topic.author_id || data.topic.can_moderate) &&
          (!data.topic.locked || data.topic.can_moderate) &&
          data.topic.status === 'published'
        "
        @click="openMetadata"
      >
        编辑标题与标签
      </button>
    </div>
    <div class="actions">
      <span v-if="data.topic.pinned" class="badge">置顶</span
      ><span v-for="tag in data.topic.tags" :key="tag" class="badge">{{
        tag
      }}</span>
    </div>
    <form v-if="metadataOpen" class="panel" @submit.prevent="saveMetadata">
      <label class="field"
        >主题标题<input
          v-model="topicTitle"
          minlength="3"
          maxlength="120"
          required /></label
      ><label class="field"
        >标签（最多 5 个，用逗号分隔）<input
          v-model="topicTags"
          maxlength="124"
      /></label>
      <div class="actions">
        <button :disabled="busy">保存主题信息</button
        ><button type="button" @click="metadataOpen = false">取消</button>
      </div>
    </form>
    <div v-if="data.topic.can_moderate" class="moderation-tools">
      <label
        >管理主题
        <select v-model="modAction">
          <option value="">选择操作</option>
          <option value="hide">隐藏</option>
          <option value="restore">恢复公开</option>
          <option value="lock">锁定</option>
          <option value="unlock">解除锁定</option>
          <option value="pin">置顶</option>
          <option value="unpin">取消置顶</option>
        </select></label
      ><template v-if="modAction"
        ><input
          v-model="modReason"
          aria-label="管理原因"
          placeholder="填写原因（至少 3 字）"
          maxlength="1000"
        /><button
          :disabled="busy || modReason.trim().length < 3"
          @click="moderate"
        >
          执行
        </button></template
      >
    </div>
    <button v-if="data.start_after" @click="load(false, true)">
      从首楼阅读
    </button>
    <TopicExtensions :data="data" @updated="load()" />
    <label class="actions"
      ><input type="checkbox" v-model="authorOnly" />只看楼主</label
    >
    <section v-if="revisions" class="panel">
      <h2>正文修订历史</h2>
      <button @click="revisions = null">关闭历史</button>
      <article v-for="r in revisions" :key="r.revision">
        <p>
          版本 {{ r.revision }} · {{ new Date(r.created_at).toLocaleString() }}
        </p>
        <Document :body="r.body" />
      </article>
      <p v-if="!revisions.length">暂无历史修订。</p>
    </section>
    <RouterLink
      v-if="reading.position.value && !route.hash"
      class="notice resume-reading"
      :to="{ hash: '#p-' + reading.position.value.post_id }"
      >继续阅读：上次到 #{{ reading.position.value.post_number }}</RouterLink
    >
    <article
      v-for="post in data.posts"
      :id="'p-' + post.id"
      :key="post.id"
      class="post"
    >
      <header class="post-header">
        <span class="avatar">{{ post.display_name.slice(0, 1) }}</span
        ><RouterLink :to="'/u/' + post.author_id"
          ><strong>{{ post.display_name }}</strong></RouterLink
        ><span v-if="post.author_id === data.topic.author_id" class="badge"
          >楼主</span
        ><a :href="'#p-' + post.id" class="muted">#{{ post.post_number }}</a
        ><time class="muted">{{
          new Date(post.created_at).toLocaleString()
        }}</time
        ><span v-if="post.edited_at" class="muted">已编辑</span>
      </header>
      <p v-if="post.blocked" class="muted">
        已屏蔽此用户的内容。可在社区设置中解除。
      </p>
      <p v-else-if="post.status === 'deleted'" class="muted">
        这条回复已删除。
      </p>
      <template v-else
        ><a v-if="post.reply_to" :href="'#p-' + post.reply_to" class="badge"
          >回复指定楼层</a
        ><template v-if="editId === post.id"
          ><label class="field"
            >修改正文<textarea
              :id="`edit-body-${post.id}`"
              v-model="editText"
              @paste="paste($event, true)"
              rows="6"
              maxlength="20000"
            />
          </label>
          <BoardSyntaxHelp @insert="insertSyntax($event, true)" />
          <RichTools @insert="insertSyntax($event, true)" />
          <details>
            <summary>预览修改</summary>
            <Document
              :body="{ blocks: [{ type: 'paragraph', text: editText }] }"
            />
          </details>
          <BoardSyntaxPreview :text="editText" />
          <div class="actions">
            <button :disabled="busy" @click="save(post)">保存修改</button
            ><button @click="editId = null">取消</button>
          </div></template
        ><Document v-else :body="post.body" />
        <div class="actions" v-if="user">
          <button
            v-if="post.author_id === user.id || data.topic.can_moderate"
            @click="history(post)"
          >
            修订历史
          </button>
          <button
            :disabled="
              busy ||
              post.author_id === user.id ||
              data.topic.locked ||
              data.topic.status !== 'published'
            "
            :aria-pressed="post.liked"
            @click="like(post)"
          >
            {{ post.liked ? "已赞同" : "赞同" }} {{ post.likes }}</button
          ><button
            :disabled="data.topic.locked || data.topic.status !== 'published'"
            @click="replyTo = post"
          >
            回复</button
          ><button
            v-if="
              (post.author_id === user.id || data.topic.can_moderate) &&
              (!data.topic.locked || data.topic.can_moderate) &&
              data.topic.status === 'published'
            "
            @click="edit(post)"
          >
            编辑</button
          ><button
            v-if="
              post.post_number > 1 &&
              (post.author_id === user.id || data.topic.can_moderate) &&
              !data.topic.locked &&
              data.topic.status === 'published'
            "
            :disabled="busy"
            @click="remove(post)"
          >
            删除</button
          ><button
            @click="
              reportPost = post.id;
              reason = '';
            "
          >
            举报
          </button>
        </div></template
      >
      <span
        class="read-marker"
        :data-post="post.id"
        :data-number="post.post_number"
        aria-hidden="true"
      ></span>
    </article>
    <button v-if="data.next_after" :disabled="loading" @click="load(true)">
      加载后续回复
    </button>
    <form v-if="reportPost" class="panel" @submit.prevent="report">
      <h2>举报内容</h2>
      <label class="field"
        >具体原因<textarea
          v-model="reason"
          rows="3"
          minlength="3"
          maxlength="1000"
          required
        />
      </label>
      <div class="actions">
        <button :disabled="busy || reason.trim().length < 3">提交举报</button
        ><button type="button" @click="reportPost = null">取消</button>
      </div>
    </form>
    <form
      v-if="user && !data.topic.locked && data.topic.status === 'published'"
      class="reply-form"
      @submit.prevent="send"
    >
      <fieldset :disabled="busy || !draft.ready.value" class="editor-fieldset">
        <h2>{{ replyTo ? "回复 #" + replyTo.post_number : "参与讨论" }}</h2>
        <button v-if="replyTo" type="button" @click="replyTo = null">
          取消指定回复</button
        ><label class="field"
          ><span class="sr-only">回复正文</span
          ><textarea
            ref="replyInput"
            v-model="reply"
            @paste="paste($event)"
            rows="5"
            maxlength="20000"
            placeholder="说说你的思路…支持 [[board:4x4:盘面编码]]"
            required
          />
        </label>
        <BoardSyntaxHelp @insert="insertSyntax($event)" />
        <RichTools @insert="insertSyntax($event)" />
        <details>
          <summary>回复预览</summary>
          <Document :body="{ blocks: [{ type: 'paragraph', text: reply }] }" />
        </details>
        <BoardSyntaxPreview :text="reply" />
        <button class="primary" :disabled="busy || !reply.trim()">
          {{ busy ? "正在提交…" : "发布回复" }}
        </button>
      </fieldset>
      <p class="muted" role="status">{{ draft.status.value }}</p>
      <p role="status">{{ pasteImage.status.value }}</p>
      <details v-if="draft.conflict.value" class="notice" open>
        <summary>云端草稿发生变化</summary>
        <pre>{{ draft.conflict.value.text || "（空草稿）" }}</pre>
        <div class="actions">
          <button type="button" @click="draft.resolve(true)">
            采用云端草稿</button
          ><button type="button" @click="draft.resolve(false)">
            用本地文本覆盖此版本
          </button>
        </div>
      </details>
      <button type="button" :disabled="busy" @click="draft.save()">
        保存回复草稿
      </button>
      <button type="button" :disabled="busy" @click="draft.restore()">
        重新同步草稿
      </button>
    </form>
    <p v-else class="notice">
      {{ !user ? "登录后可以参与讨论。" : "此主题目前不能回复。" }}
    </p>
    <p role="status" class="status">{{ status }}</p></template
  >
</template>
