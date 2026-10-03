<script setup>
import { ref } from "vue";
import { api, uploadAsset } from "../api";
import MentionPicker from "./MentionPicker.vue";
const emit = defineEmits(["insert"]);
const busy = ref(false),
  error = ref(""),
  runId = ref("");
const tools = [
  ["粗体", "**重点内容**"],
  ["斜体", "*强调内容*"],
  ["标题", "## 小标题"],
  ["引用", "> 引用内容"],
  ["列表", "- 第一项\n- 第二项"],
  ["代码", "```text\n代码或棋盘语法原文\n```"],
  [
    "折叠文本",
    "```details 点击展开说明\n这里的内容按纯文本显示，不触发提及或附件。\n```",
  ],
  ["链接", "[链接文字](https://2048tables.online/)"],
];
async function upload(event, kind) {
  const file = event.target.files?.[0];
  if (!file) return;
  error.value = "";
  busy.value = true;
  try {
    if (file.size > (kind === "image" ? 5 : 2) * 1024 * 1024)
      throw Error(kind === "image" ? "图片最多 5 MiB。" : "录像最多 2 MiB。");
    const result = await uploadAsset(file, kind);
    emit(
      "insert",
      kind === "image"
        ? `![图片说明](/api/forum/v1/media/${result.id})`
        : `[[replay:${result.id}]]`,
    );
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
    event.target.value = "";
  }
}
async function importPlay() {
  error.value = "";
  busy.value = true;
  try {
    const id = runId.value.trim();
    if (!/^[0-9a-f-]{36}$/i.test(id))
      throw Error("请输入 Play 对局 ID（UUID）。");
    const result = await api("/media/play/" + id, { method: "POST" });
    emit("insert", `[[replay:${result.id}]]`);
    runId.value = "";
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
</script>
<template>
  <div class="rich-tools">
    <div class="actions" role="toolbar" aria-label="正文格式">
      <button
        v-for="[label, syntax] in tools"
        :key="label"
        type="button"
        :disabled="busy"
        @click="emit('insert', syntax)"
      >
        {{ label }}
      </button>
    </div>
    <details>
      <summary>插入图片或录像</summary>
      <p class="muted">
        图片支持 PNG / JPEG / WebP（5 MiB）；录像支持 Play HPR、Verse / RPL1（2
        MiB）。附件发布前仅自己可见。
      </p>
      <label class="field"
        >上传图片<input
          type="file"
          accept="image/png,image/jpeg,image/webp"
          :disabled="busy"
          @change="upload($event, 'image')"
      /></label>
      <label class="field"
        >上传录像<input
          type="file"
          accept=".hpr,.vrs,.rpl,.rpl1,.txt,.gz,.fbr"
          :disabled="busy"
          @change="upload($event, 'replay')"
      /></label>
      <label class="field"
        >引用公开 Play 对局<input
          v-model="runId"
          placeholder="Play 对局 ID"
          :disabled="busy" /></label
      ><button
        type="button"
        :disabled="busy || !runId.trim()"
        @click="importPlay"
      >
        插入公开录像
      </button>
      <p class="muted">
        本地录像仅校验规则，不作为正式成绩认证。Play 来源撤回后会停止提供回放。
      </p>
    </details>
    <MentionPicker @insert="emit('insert', $event)" />
    <p v-if="busy" role="status">正在上传并校验…</p>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
  </div>
</template>
