<script setup>
import { ref, computed, onBeforeUnmount, watch } from "vue";
import { api } from "../api";
import Board from "./Board.vue";
const props = defineProps({
  id: String,
  initialStep: { type: Number, default: 0 },
});
const loading = ref(false),
  error = ref(""),
  data = ref(null),
  frame = ref(null),
  total = ref(0),
  step = ref(0),
  playing = ref(false),
  speed = ref(250),
  status = ref("");
let worker,
  timer,
  controller,
  epoch = 0;
const board = computed(() => {
  if (!frame.value) return null;
  const [rows, cols] = data.value.variant.split("x").map(Number);
  return {
    type: "board",
    rows,
    cols,
    cells: frame.value.board,
    source: "replay",
    caption: "",
  };
});
function stop() {
  clearInterval(timer);
  playing.value = false;
}
function seek(value) {
  stop();
  worker?.postMessage({ type: "seek", step: Number(value) });
}
function cleanup() {
  epoch++;
  controller?.abort();
  worker?.terminate();
  worker = null;
  stop();
  data.value = null;
  frame.value = null;
}
async function load() {
  cleanup();
  const ticket = epoch;
  controller = new AbortController();
  loading.value = true;
  error.value = "";
  try {
    const result = await api("/media/" + props.id + "/replay", {
      signal: controller.signal,
    });
    if (ticket !== epoch) return;
    data.value = result;
    worker = new Worker(new URL("../replay.worker.js", import.meta.url), {
      type: "module",
    });
    worker.onerror = () => {
      error.value = "回放加载失败，请重试。";
      loading.value = false;
      stop();
    };
    worker.onmessage = ({ data: message }) => {
      if (ticket !== epoch) return;
      if (message.type === "ready") {
        total.value = message.total;
        loading.value = false;
      }
      if (message.type === "frame") {
        frame.value = message;
        step.value = message.step;
        if (step.value >= total.value) stop();
      }
      if (message.type === "error") {
        error.value = message.message;
        loading.value = false;
        stop();
      }
    };
    worker.postMessage({
      type: "load",
      replay: result,
      step: props.initialStep,
    });
  } catch (e) {
    if (ticket === epoch) {
      error.value = e.message;
      loading.value = false;
    }
  }
}
function toggle() {
  if (playing.value) {
    stop();
    return;
  }
  if (step.value >= total.value) seek(0);
  playing.value = true;
  timer = setInterval(
    () =>
      worker?.postMessage({
        type: "seek",
        step: Math.min(total.value, step.value + 1),
      }),
    Number(speed.value),
  );
}
async function copy() {
  const text = `[[replay:${props.id}@${step.value}]]`;
  try {
    await navigator.clipboard.writeText(text);
    status.value = "当前步数语法已复制";
  } catch {
    status.value = text;
  }
}
watch(
  () => props.id,
  () => {
    cleanup();
    loading.value = false;
    error.value = "";
  },
);
onBeforeUnmount(cleanup);
</script>
<template>
  <section class="replay-player panel">
    <div class="heading">
      <strong>对局回放</strong
      ><button type="button" :disabled="loading" @click="load">
        {{ loading ? "正在校验…" : data ? "重新读取来源" : "加载录像" }}
      </button>
    </div>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    <template v-if="data"
      ><p class="muted">
        {{ data.variant }} ·
        {{
          data.verification === "public-source"
            ? "Play 公开来源，已重新检查可见性"
            : "用户上传，规则校验通过；非正式成绩"
        }}
        <a
          v-if="data.source_url"
          :href="data.source_url"
          target="_blank"
          rel="noopener"
          >源录像</a
        >
      </p>
      <Board v-if="board" :board="board" />
      <p v-if="frame">
        第 {{ step }} / {{ total }} 步 · 分数
        {{ frame.score.toLocaleString() }} ·
        {{ (frame.elapsed / 1000).toFixed(2) }} 秒（已知时长）
      </p>
      <label class="field"
        >跳转步数<input
          type="range"
          :min="0"
          :max="total"
          :value="step"
          :disabled="loading"
          @input="seek($event.target.value)"
      /></label>
      <div class="actions">
        <button
          type="button"
          :disabled="loading || !step"
          @click="seek(step - 1)"
        >
          上一步</button
        ><button type="button" :disabled="loading || !total" @click="toggle">
          {{ playing ? "暂停" : "播放" }}</button
        ><button
          type="button"
          :disabled="loading || step >= total"
          @click="seek(step + 1)"
        >
          下一步
        </button>
        <select v-model="speed" aria-label="回放速度" @change="stop">
          <option :value="500">慢速</option>
          <option :value="250">标准</option>
          <option :value="100">快速</option></select
        ><button type="button" @click="copy">复制当前步数</button>
      </div>
      <p class="muted" role="status">{{ status }}</p></template
    >
    <p v-else-if="!loading" class="muted">按需加载，支持逐步查看和定位讨论。</p>
  </section>
</template>
