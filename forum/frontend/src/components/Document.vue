<script setup>
import { computed } from "vue";
import { renderDocumentBlocks } from "../boardSyntax";
import Board from "./Board.vue";
import MarkdownContent from "./MarkdownContent";
import ReplayPlayer from "./ReplayPlayer.vue";
const props = defineProps({ body: Object });
const blocks = computed(() => renderDocumentBlocks(props.body));
</script>
<template>
  <div class="document">
    <template v-for="(block, i) in blocks" :key="i"
      ><MarkdownContent v-if="block.type === 'paragraph'" :text="block.text" />
      <Board v-else-if="block.type === 'board'" :board="block" />
      <ReplayPlayer
        v-else-if="block.type === 'replay'"
        :id="block.id"
        :initial-step="block.step"
      />
      <p v-else class="muted">此内容需要更新的阅读器。</p></template
    >
  </div>
</template>
