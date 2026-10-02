<script setup>
import { computed } from "vue";
import { renderDocumentBlocks } from "../boardSyntax";
import Board from "./Board.vue";
const props = defineProps({ body: Object });
const blocks = computed(() => renderDocumentBlocks(props.body));
</script>
<template>
  <div class="document">
    <template v-for="(block, i) in blocks" :key="i"
      ><p v-if="block.type === 'paragraph'" class="prose">{{ block.text }}</p>
      <Board v-else-if="block.type === 'board'" :board="block" />
      <p v-else class="muted">此内容需要更新的阅读器。</p></template
    >
  </div>
</template>
