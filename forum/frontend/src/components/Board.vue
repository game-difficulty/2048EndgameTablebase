<script setup>
import { computed, ref } from "vue";
import { encodeBoardSyntax } from "../boardSyntax";
import { tileStyle } from "../tilePalette";
const props = defineProps({
  board: { type: Object, required: true },
  editable: Boolean,
});
defineEmits(["cell"]);
const syntax = computed(() => encodeBoardSyntax(props.board));
const copyStatus = ref("");
const showSource = ref(false);
async function copy() {
  try {
    await navigator.clipboard.writeText(syntax.value);
    copyStatus.value = "语法已复制";
  } catch {
    showSource.value = true;
    copyStatus.value = "请选中下方语法手动复制";
  }
}
function label(value) {
  return value >= 1024 ? `${value / 1024}K` : value || "";
}
</script>
<template>
  <figure class="board-figure">
    <div
      class="board-grid"
      :style="{ gridTemplateColumns: `repeat(${board.cols},minmax(0,1fr))` }"
      :aria-label="`${board.rows} 行 ${board.cols} 列${board.source ? '编码' : '手绘'}局面`"
    >
      <template v-for="(value, i) in board.cells" :key="i"
        ><button
          v-if="editable"
          type="button"
          class="tile"
          :style="tileStyle(value)"
          :aria-label="`第 ${Math.floor(i / board.cols) + 1} 行第 ${(i % board.cols) + 1} 列，${value || '空'}，点击绘制`"
          @click="$emit('cell', i)"
        >
          {{ label(value) }}
        </button>
        <div
          v-else
          class="tile"
          :style="tileStyle(value)"
          :aria-label="String(value || '空')"
        >
          {{ label(value) }}
        </div></template
      >
    </div>
    <figcaption>
      <span class="badge"
        >{{ board.rows }}×{{ board.cols }} ·
        {{
          board.source === "replay"
            ? "录像局面"
            : board.source
              ? "编码局面"
              : "手绘局面"
        }}
        · 非正式成绩</span
      >
      <p v-if="board.caption">{{ board.caption }}</p>
      <div v-if="syntax" class="board-source">
        <button type="button" @click="copy">复制棋盘语法</button>
        <button
          type="button"
          :aria-expanded="showSource"
          @click="showSource = !showSource"
        >
          {{ showSource ? "收起语法" : "查看语法" }}
        </button>
        <span role="status">{{ copyStatus }}</span>
        <code v-if="showSource">{{ syntax }}</code>
      </div>
    </figcaption>
  </figure>
</template>
