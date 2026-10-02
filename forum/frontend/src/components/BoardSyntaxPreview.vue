<script setup>
import { computed } from "vue";
import { parseBoardText } from "../boardSyntax";
import Document from "./Document.vue";
const props = defineProps({ text: { type: String, default: "" } });
const parsed = computed(() => parseBoardText(props.text));
</script>
<template>
  <section
    v-if="parsed.boards || parsed.diagnostics.length"
    class="syntax-preview"
    aria-label="棋盘语法预览"
  >
    <p class="muted">发布效果 · {{ parsed.boards }} 个棋盘</p>
    <ul v-if="parsed.diagnostics.length" class="syntax-errors" role="status">
      <li v-for="(issue, index) in parsed.diagnostics" :key="index">
        第 {{ text.slice(0, issue.offset).split("\n").length }} 行：{{
          issue.message
        }}
        未识别的内容会保留原文。
      </li>
    </ul>
    <Document :body="{ blocks: [{ type: 'paragraph', text }] }" />
  </section>
</template>
