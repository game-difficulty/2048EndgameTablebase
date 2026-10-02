<script setup>
import { BOARD_EXAMPLE } from "../boardSyntax";
defineEmits(["insert"]);
</script>
<template>
  <details class="syntax-help">
    <summary>用语法插入棋盘</summary>
    <p>直接粘贴 Play 的盘面编码，无需逐格绘制。发帖和回复均支持。</p>
    <code>{{ BOARD_EXAMPLE }}</code>
    <p>支持 4x4、3x4、3x3、2x4；按从左到右、从上到下排列。可用 / 分行。</p>
    <p>
      0 = 空格，1 = 2，2 = 4，a = 1024，f = 32768，g = 65536；最高 k =
      1048576。大小写均可，连续短码在左侧补 0。
    </p>
    <p>| 后是可选说明。用反引号包住语法，或在 [[ 前加反斜杠，可显示原文。</p>
    <button type="button" @click="$emit('insert', BOARD_EXAMPLE)">
      插入棋盘示例
    </button>
  </details>
</template>
