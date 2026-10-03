<script setup>
import { computed, ref } from "vue";
import { move, VARIANTS } from "../../../../frontend/src/human/engine.js";
import { uploadAsset } from "../api";
import Board from "./Board.vue";
const props = defineProps({ initial: Array, variant: String, origin: String }),
  emit = defineEmits(["close"]);
const cells = ref([...props.initial]),
  moves = ref([]),
  past = ref([]),
  pending = ref(null),
  spawn = ref(2),
  error = ref(""),
  busy = ref(false),
  syntax = ref("");
const board = computed(() => ({
  rows: VARIANTS[props.variant][0],
  cols: VARIANTS[props.variant][1],
  cells: cells.value,
  source: "branch",
  caption: "假设分支 · 手动指定出数 · 非正式成绩",
}));
function step(direction) {
  if (moves.value.length >= 1000) return;
  const next = move(cells.value, ...VARIANTS[props.variant], direction);
  if (!next.changed) {
    error.value = "这个方向无法移动。";
    return;
  }
  error.value = "";
  pending.value = { direction, before: [...cells.value] };
  cells.value = next.board;
  syntax.value = "";
}
function place(index) {
  if (!pending.value || cells.value[index]) return;
  past.value.push(pending.value.before);
  cells.value[index] = Number(spawn.value);
  moves.value.push([pending.value.direction, index, Number(spawn.value), 0]);
  pending.value = null;
}
function undo() {
  if (pending.value) {
    cells.value = pending.value.before;
    pending.value = null;
  } else if (past.value.length) {
    cells.value = past.value.pop();
    moves.value.pop();
  }
  syntax.value = "";
}
async function save() {
  busy.value = true;
  error.value = "";
  try {
    const body = {
      variant: props.variant,
      initial: props.initial,
      moves: moves.value,
      origin: props.origin || "",
    };
    const file = new File(["FBR1" + JSON.stringify(body)], "hypothetical.fbr", {
      type: "application/octet-stream",
    });
    const r = await uploadAsset(file, "replay");
    syntax.value = `[[replay:${r.id}]]`;
    try {
      await navigator.clipboard.writeText(syntax.value);
    } catch {}
  } catch (e) {
    error.value = e.message;
  } finally {
    busy.value = false;
  }
}
</script>
<template>
  <section class="panel">
    <h3>从当前局面试走</h3>
    <p class="muted">
      每次移动后，选择生成的 2 或 4，再点一个空格。完整记录出数，最多 1000
      步。保存为假设分支，可复制语法到回复。
    </p>
    <Board :board="board" :editable="!!pending" @cell="place" />
    <fieldset class="editor-fieldset" :disabled="busy">
      <div class="actions">
        <button
          v-for="(name, d) in ['上', '右', '下', '左']"
          :key="d"
          :disabled="!!pending || moves.length >= 1000"
          @click="step(d)"
        >
          {{ name }}</button
        ><label
          >生成块<select v-model="spawn">
            <option :value="2">2</option>
            <option :value="4">4</option>
          </select></label
        ><button :disabled="!pending && !moves.length" @click="undo">
          撤销一步
        </button>
      </div>
      <p>
        {{
          pending ? "请选择一个空格生成小块" : "已记录 " + moves.length + " 步"
        }}
      </p>
      <div class="actions">
        <button :disabled="!!pending || !moves.length" @click="save">
          保存并复制分支语法</button
        ><button @click="emit('close')">关闭试走</button>
      </div>
    </fieldset>
    <p v-if="error" class="notice error">{{ error }}</p>
    <code v-if="syntax">{{ syntax }}</code>
  </section>
</template>
