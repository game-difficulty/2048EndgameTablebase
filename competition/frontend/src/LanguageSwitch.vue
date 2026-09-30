<script setup>
import { languageOverride, setLanguage, t } from './i18n.js';
</script>
<template><div class="competition-language" aria-label="Language / 语言"><button type="button" :title="t('跟随主站语言，仅同步到本站')" :aria-label="t('跟随主站语言，仅同步到本站')" :aria-pressed="!languageOverride" @click="setLanguage('auto')">Auto</button><button type="button" :aria-pressed="languageOverride==='zh'" @click="setLanguage('zh')">中文</button><button type="button" :aria-pressed="languageOverride==='en'" @click="setLanguage('en')">EN</button></div></template>
<style scoped>
.competition-language{display:inline-flex;gap:2px;align-items:center;border:1px solid currentColor;border-radius:5px;padding:2px;font-size:12px;flex-shrink:0}.competition-language button{border:0;background:transparent;color:inherit;padding:4px 7px;border-radius:3px;font:inherit;cursor:pointer}.competition-language button[aria-pressed=true]{background:#9f7b3e;color:white}
</style>
