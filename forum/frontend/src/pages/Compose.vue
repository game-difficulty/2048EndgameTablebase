<script setup>
import { computed, inject, nextTick, onBeforeUnmount, ref, watch } from "vue";
import { onBeforeRouteLeave, useRoute, useRouter } from "vue-router";
import { api, submission } from "../api";
import Board from "../components/Board.vue";
import Document from "../components/Document.vue";
import BoardSyntaxHelp from "../components/BoardSyntaxHelp.vue";
import BoardSyntaxPreview from "../components/BoardSyntaxPreview.vue";
import { insertAtCursor } from "../editorInsertion";
import { refreshTilePalette } from "../tilePalette";
import RichTools from "../components/RichTools.vue";
import { usePasteImage } from "../pasteImage";
const bodyInput = ref(null);
function insertSyntax(snippet) {
  insertAtCursor(text, bodyInput.value, snippet);
}
const { user, boards } = inject("forum"),
  route = useRoute(),
  router = useRouter();
const title = ref(""),
  kind = ref("discussion"),
  tags = ref(""),
  pollOptions = ref(""),
  pollDeadline = ref(""),
  pollMax = ref(1),
  pollResults = ref("always"),
  text = ref(""),
  boardSlug = ref("general"),
  board = ref(null),
  pen = ref(2),
  history = ref([]),
  size = ref("4x4");
const preview = ref(false),
  error = ref(""),
  status = ref(""),
  publishing = ref(false),
  saving = ref(false),
  drafts = ref([]);
const draftId = ref(crypto.randomUUID()),
  revision = ref(0),
  lastSaved = ref("");
const key = submission();
const pasteImage = usePasteImage(() => `${user.value?.id}:${draftId.value}`);
function paste(event) {
  pasteImage.paste(event, text);
}
let timer = null,
  savePromise = null,
  restoring = false,
  conflict = false,
  epoch = 0;
const allowed = computed(() => boards.value.filter((b) => b.can_post));
const payload = computed(() => ({
  title: title.value,
  board_slug: boardSlug.value,
  text: text.value,
  board: board.value,
  kind: kind.value,
  tags: tags.value
    .split(/[、,，]/)
    .map((x) => x.trim())
    .filter(Boolean),
  poll:
    kind.value === "poll"
      ? {
          options: pollOptions.value,
          closes_at: pollDeadline.value,
          max_choices: Number(pollMax.value),
          results: pollResults.value,
        }
      : null,
}));
const signature = computed(() => JSON.stringify(payload.value));
const dirty = computed(
  () =>
    signature.value !== lastSaved.value &&
    (!!title.value || !!text.value || !!board.value),
);
const body = computed(() => ({
  version: 1,
  blocks: [
    ...(text.value.trim()
      ? [{ type: "paragraph", text: text.value.trim() }]
      : []),
    ...(board.value ? [board.value] : []),
  ],
}));
function localKey(id) {
  return "forum:draft:" + id;
}
function remember() {
  if (!user.value) return;
  try {
    localStorage.setItem(
      localKey(user.value.id),
      JSON.stringify({
        id: draftId.value,
        revision: revision.value,
        payload: payload.value,
      }),
    );
  } catch {}
}
function restore(draft) {
  restoring = true;
  draftId.value = draft.id;
  revision.value = draft.revision;
  title.value = draft.payload.title || "";
  text.value = draft.payload.text || "";
  kind.value = draft.payload.kind || "discussion";
  tags.value = (draft.payload.tags || []).join("、");
  pollOptions.value = draft.payload.poll?.options || "";
  pollDeadline.value = draft.payload.poll?.closes_at || "";
  pollMax.value = draft.payload.poll?.max_choices || 1;
  pollResults.value = draft.payload.poll?.results || "always";
  boardSlug.value = draft.payload.board_slug || "general";
  board.value = draft.payload.board
    ? JSON.parse(JSON.stringify(draft.payload.board))
    : null;
  if (board.value) size.value = `${board.value.rows}x${board.value.cols}`;
  history.value = [];
  conflict = false;
  error.value = "";
  lastSaved.value = signature.value;
  nextTick(() => (restoring = false));
}
async function listDrafts() {
  const identity = user.value?.id;
  if (!identity) return;
  const result = await api("/drafts");
  if (user.value?.id === identity) drafts.value = result.items;
}
async function save() {
  if (savePromise) {
    await savePromise;
    if (dirty.value && !conflict) return save();
    return;
  }
  if (!user.value || !dirty.value || conflict) return;
  const ticket = epoch,
    identity = user.value.id,
    snapshot = JSON.parse(signature.value),
    snapshotText = signature.value,
    id = draftId.value;
  saving.value = true;
  const pending = (async () => {
    try {
      const result = await api("/drafts/" + id, {
        method: "PUT",
        body: { ...snapshot, revision: revision.value },
      });
      if (ticket !== epoch || identity !== user.value?.id) return;
      revision.value = result.revision;
      lastSaved.value = snapshotText;
      status.value = "草稿已保存";
      remember();
    } catch (e) {
      if (ticket === epoch) {
        error.value = e.message;
        conflict = e.code === "REVISION_CONFLICT";
        status.value = "未同步，内容保留在本机";
      }
    } finally {
      if (ticket === epoch) saving.value = false;
    }
  })();
  savePromise = pending;
  await pending;
  if (savePromise === pending) savePromise = null;
}
watch(
  payload,
  () => {
    if (restoring || !user.value) return;
    remember();
    clearTimeout(timer);
    timer = setTimeout(save, 1200);
  },
  { deep: true },
);
watch(
  () => user.value?.id,
  async (id) => {
    epoch++;
    clearTimeout(timer);
    conflict = false;
    saving.value = false;
    savePromise = null;
    drafts.value = [];
    restore({
      id: crypto.randomUUID(),
      revision: 0,
      payload: { board_slug: route.query.board || "general" },
    });
    if (!id) return;
    try {
      await listDrafts();
      if (user.value?.id !== id) return;
      const local = JSON.parse(localStorage.getItem(localKey(id)) || "null");
      if (local?.payload && local.id) {
        restore(local);
        lastSaved.value = "";
        status.value = "已恢复本机草稿";
      }
    } catch (e) {
      error.value = e.message;
    }
  },
  { immediate: true },
);
async function openDraft(d) {
  if (dirty.value) {
    await save();
    if (dirty.value && !confirm("当前内容尚未同步，仍要打开另一份草稿吗？"))
      return;
  }
  restore(d);
  remember();
}
function copyDraft() {
  draftId.value = crypto.randomUUID();
  revision.value = 0;
  lastSaved.value = "";
  conflict = false;
  error.value = "";
  save();
}
function addBoard() {
  const [rows, cols] = size.value.split("x").map(Number);
  history.value.push(
    board.value ? JSON.parse(JSON.stringify(board.value)) : null,
  );
  board.value = {
    type: "board",
    rows,
    cols,
    cells: Array(rows * cols).fill(0),
    caption: "",
  };
}
function resize() {
  if (
    board.value?.cells.some(Boolean) &&
    !confirm("更改尺寸会清空当前棋盘，是否继续？")
  ) {
    size.value = `${board.value.rows}x${board.value.cols}`;
    return;
  }
  addBoard();
}
function paint(i) {
  history.value.push(JSON.parse(JSON.stringify(board.value)));
  if (history.value.length > 100) history.value.shift();
  board.value.cells[i] = pen.value;
}
function undo() {
  if (history.value.length) board.value = history.value.pop();
}
function exportPng() {
  refreshTilePalette();
  const colors = getComputedStyle(document.documentElement);
  const b = board.value,
    c = document.createElement("canvas"),
    cell = 140,
    gap = 12;
  c.width = b.cols * cell + gap * 2;
  c.height = b.rows * cell + 116;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#faf7f0";
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  b.cells.forEach((v, i) => {
    const x = gap + (i % b.cols) * cell,
      y = gap + Math.floor(i / b.cols) * cell;
    ctx.fillStyle = v
      ? colors.getPropertyValue(`--color-tile-${v}`).trim() || "#000000"
      : "#cdc1b4";
    ctx.fillRect(x, y, cell - gap, cell - gap);
    ctx.fillStyle = v
      ? colors.getPropertyValue(`--color-text-${v}`).trim() || "#f9f6f2"
      : "#776e65";
    ctx.font = `bold ${v >= 100000 ? 26 : 32}px sans-serif`;
    if (v) ctx.fillText(String(v), x + (cell - gap) / 2, y + (cell - gap) / 2);
  });
  ctx.font = "20px sans-serif";
  ctx.fillStyle = "#433528";
  ctx.fillText("2048 社区 · 手绘局面 · 非正式成绩", c.width / 2, c.height - 52);
  c.toBlob((blob) => {
    if (!blob) return;
    const url = URL.createObjectURL(blob),
      a = document.createElement("a");
    a.href = url;
    a.download = "2048-board.png";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, "image/png");
}
async function publish() {
  if (publishing.value) return;
  publishing.value = true;
  error.value = "";
  clearTimeout(timer);
  try {
    await save();
    const request = {
      board_slug: boardSlug.value,
      title: title.value,
      body: body.value,
      tags: payload.value.tags,
      kind: kind.value,
      poll:
        kind.value === "poll"
          ? {
              options: pollOptions.value
                .split("\n")
                .map((x) => x.trim())
                .filter(Boolean),
              closes_at: new Date(pollDeadline.value).toISOString(),
              max_choices: Number(pollMax.value),
              results: pollResults.value,
            }
          : null,
    };
    const result = await api("/topics", {
      method: "POST",
      body: request,
      key: key(request),
    });
    const oldId = draftId.value,
      oldRev = revision.value;
    lastSaved.value = signature.value;
    try {
      localStorage.removeItem(localKey(user.value.id));
    } catch {}
    if (oldRev) {
      try {
        await api(`/drafts/${oldId}?revision=${oldRev}`, { method: "DELETE" });
      } catch {}
    }
    await router.push("/t/" + result.topic_id);
  } catch (e) {
    error.value = e.message;
    remember();
  } finally {
    publishing.value = false;
  }
}
function unload(e) {
  if (dirty.value) {
    e.preventDefault();
    e.returnValue = "";
  }
}
window.addEventListener("beforeunload", unload);
onBeforeRouteLeave(async () => {
  clearTimeout(timer);
  await save();
  return !dirty.value || confirm("内容尚未同步，已保留本机副本。仍要离开吗？");
});
onBeforeUnmount(() => {
  epoch++;
  clearTimeout(timer);
  window.removeEventListener("beforeunload", unload);
});
</script>
<template>
  <div v-if="!user" class="empty">
    <h1>登录后开始创作</h1>
    <a class="button primary" href="https://play.2048tables.online/"
      >前往 Play 登录</a
    >
  </div>
  <template v-else>
    <fieldset class="composer-fields" :disabled="publishing">
      <div class="panel">
        <label class="field"
          >主题类型<select v-model="kind">
            <option value="discussion">讨论 / 作品</option>
            <option value="question">提问与反馈</option>
            <option value="poll">投票</option>
          </select></label
        >
        <label class="field"
          >标签（逗号分隔，最多 5 个）<input v-model="tags" maxlength="124"
        /></label>
        <template v-if="kind === 'poll'"
          ><label class="field"
            >投票选项（每行一项，2—10 项）<textarea
              v-model="pollOptions"
              maxlength="1300"
              rows="4"
            /></label
          ><label class="field"
            >投票截止时间<input
              v-model="pollDeadline"
              type="datetime-local"
              required /></label
          ><label class="field"
            >每人最多选择<input
              v-model="pollMax"
              type="number"
              min="1"
              max="10" /></label
          ><label class="field"
            >结果显示<select v-model="pollResults">
              <option value="always">始终显示</option>
              <option value="voted">投票后显示</option>
              <option value="closed">截止后显示</option>
            </select></label
          >
          <p class="muted">
            发布后选项固定；截止前可以修改自己的选择。
          </p></template
        >
      </div>
      <div class="heading">
        <div>
          <p class="eyebrow">CREATE & DISCUSS</p>
          <h1>把你的思路写下来</h1>
          <p class="muted">从一个问题、一段经历或一个棋盘开始。</p>
        </div>
        <button @click="preview = !preview">
          {{ preview ? "继续编辑" : "预览" }}
        </button>
      </div>
      <div v-if="error" class="notice error" role="alert">
        {{ error }} <button @click="save">重试保存</button
        ><button v-if="conflict" @click="copyDraft">另存为新草稿</button>
      </div>
      <details
        class="draft-list"
        @toggle="
          $event.target.open && listDrafts().catch((e) => (error = e.message))
        "
      >
        <summary>我的草稿（{{ drafts.length }}）</summary>
        <div v-if="!drafts.length" class="muted">暂时没有云端草稿。</div>
        <div v-for="d in drafts" :key="d.id" class="actions">
          <button @click="openDraft(d)">
            {{ d.payload.title || "未命名草稿" }}</button
          ><span class="muted">{{
            new Date(d.updated_at).toLocaleString()
          }}</span>
        </div>
      </details>
      <div v-if="preview" class="panel">
        <h2>{{ title || "未命名主题" }}</h2>
        <Document :body="body" />
      </div>
      <form v-else @submit.prevent="publish">
        <label class="field"
          >标题<input
            v-model="title"
            minlength="3"
            maxlength="120"
            required
            placeholder="一个清楚的标题，让讨论更容易开始" /></label
        ><label class="field"
          >发布板块<select v-model="boardSlug">
            <option v-for="b in allowed" :key="b.id" :value="b.slug">
              {{ b.name }}
            </option>
          </select></label
        ><label class="field"
          >正文<textarea
            ref="bodyInput"
            v-model="text"
            @paste="paste"
            rows="9"
            maxlength="20000"
            placeholder="描述你的发现、问题和想法…也可输入 [[board:4x4:盘面编码]] 插入棋盘"
          />
        </label>
        <BoardSyntaxHelp @insert="insertSyntax" />
        <RichTools @insert="insertSyntax" />
        <p role="status">{{ pasteImage.status.value }}</p>
        <BoardSyntaxPreview :text="text" />
        <div class="actions">
          <button type="button" @click="board ? (board = null) : addBoard()">
            {{ board ? "移除棋盘" : "＋ 绘制局面" }}
          </button>
        </div>
        <div v-if="board" class="board-editor panel">
          <div class="heading">
            <h2>局面编辑器</h2>
            <label
              >棋盘
              <select v-model="size" @change="resize">
                <option>4x4</option>
                <option>3x4</option>
                <option>3x3</option>
                <option>2x4</option>
              </select></label
            >
          </div>
          <Board :board="board" editable @cell="paint" />
          <p class="muted">选择画笔数值，再点格子。高位方块用 K 简写。</p>
          <div class="palette">
            <button
              v-for="n in [
                0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                16384, 32768, 65536,
              ]"
              :key="n"
              type="button"
              :aria-pressed="pen === n"
              @click="pen = n"
            >
              {{ n || "擦除" }}
            </button>
          </div>
          <div class="actions">
            <button type="button" :disabled="!history.length" @click="undo">
              撤销</button
            ><button type="button" @click="exportPng">导出 PNG</button>
          </div>
          <label class="field"
            >棋盘说明<input
              v-model="board.caption"
              maxlength="500"
              placeholder="说明关键位置或想比较的走法"
          /></label>
        </div>
        <div class="actions publish-actions">
          <button
            class="primary"
            :disabled="
              publishing ||
              title.trim().length < 3 ||
              !body.blocks.length ||
              !allowed.some((b) => b.slug === boardSlug)
            "
          >
            {{ publishing ? "正在发布…" : "发布主题" }}</button
          ><button type="button" :disabled="saving" @click="save">
            保存草稿</button
          ><span class="muted" role="status">{{
            saving ? "正在保存…" : dirty ? "有未同步的修改" : status
          }}</span>
        </div>
      </form>
    </fieldset>
  </template>
</template>
<style scoped>
.composer-fields {
  border: 0;
  padding: 0;
  margin: 0;
  min-width: 0;
}
</style>
