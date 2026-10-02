<script setup>
import { inject, nextTick, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import { api, submission } from "../api";
import Document from "../components/Document.vue";
import BoardSyntaxHelp from "../components/BoardSyntaxHelp.vue";
import BoardSyntaxPreview from "../components/BoardSyntaxPreview.vue";
import { insertAtCursor } from "../editorInsertion";
const replyInput = ref(null);
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
    const result = await api(`/topics/${route.params.id}${query}`);
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
    reply.value = "";
    replyTo.value = null;
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
  () => [route.params.id, route.hash, user.value?.id],
  () => {
    data.value = null;
    reply.value = "";
    replyTo.value = null;
    editId.value = null;
    reportPost.value = null;
    load();
  },
  { immediate: true },
);
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
    </div>
    <div v-if="data.topic.can_moderate" class="moderation-tools">
      <label
        >管理主题
        <select v-model="modAction">
          <option value="">选择操作</option>
          <option value="hide">隐藏</option>
          <option value="restore">恢复公开</option>
          <option value="lock">锁定</option>
          <option value="unlock">解除锁定</option>
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
    <article
      v-for="post in data.posts"
      :id="'p-' + post.id"
      :key="post.id"
      class="post"
    >
      <header class="post-header">
        <span class="avatar">{{ post.display_name.slice(0, 1) }}</span
        ><strong>{{ post.display_name }}</strong
        ><span v-if="post.author_id === data.topic.author_id" class="badge"
          >楼主</span
        ><a :href="'#p-' + post.id" class="muted">#{{ post.post_number }}</a
        ><time class="muted">{{
          new Date(post.created_at).toLocaleString()
        }}</time
        ><span v-if="post.edited_at" class="muted">已编辑</span>
      </header>
      <p v-if="post.status === 'deleted'" class="muted">这条回复已删除。</p>
      <template v-else
        ><a v-if="post.reply_to" :href="'#p-' + post.reply_to" class="badge"
          >回复指定楼层</a
        ><template v-if="editId === post.id"
          ><label class="field"
            >修改正文<textarea
              :id="`edit-body-${post.id}`"
              v-model="editText"
              rows="6"
              maxlength="20000"
            />
          </label>
          <BoardSyntaxHelp @insert="insertSyntax($event, true)" />
          <BoardSyntaxPreview :text="editText" />
          <div class="actions">
            <button :disabled="busy" @click="save(post)">保存修改</button
            ><button @click="editId = null">取消</button>
          </div></template
        ><Document v-else :body="post.body" />
        <div class="actions" v-if="user">
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
              !data.topic.locked &&
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
      <h2>{{ replyTo ? "回复 #" + replyTo.post_number : "参与讨论" }}</h2>
      <button v-if="replyTo" type="button" @click="replyTo = null">
        取消指定回复</button
      ><label class="field"
        ><span class="sr-only">回复正文</span
        ><textarea
          ref="replyInput"
          v-model="reply"
          rows="5"
          maxlength="20000"
          placeholder="说说你的思路…支持 [[board:4x4:盘面编码]]"
          required
        />
      </label>
      <BoardSyntaxHelp @insert="insertSyntax($event)" />
      <BoardSyntaxPreview :text="reply" />
      <button class="primary" :disabled="busy || !reply.trim()">
        {{ busy ? "正在提交…" : "发布回复" }}
      </button>
    </form>
    <p v-else class="notice">
      {{ !user ? "登录后可以参与讨论。" : "此主题目前不能回复。" }}
    </p>
    <p role="status" class="status">{{ status }}</p></template
  >
</template>
