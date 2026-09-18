<template>
  <main class="more-page">
    <h1>{{ $t('menu.more') }}</h1>
    <section v-for="group in AUXILIARY_GROUPS" :key="group.id">
      <h2>{{ $t(group.title) }}</h2>
      <div class="more-entries">
        <AuxiliaryEntry v-for="entry in group.entries" :key="entry.id" :entry="entry" @navigate="$emit('navigate', $event)" />
      </div>
    </section>
  </main>
</template>
<script setup>
import AuxiliaryEntry from '../../components/AuxiliaryEntry.vue';
import { AUXILIARY_GROUPS } from '../../app/auxiliaryEntries.js';
defineEmits(['navigate']);
</script>
<style scoped>
.more-page { max-width:960px;height:100%;margin:0 auto;padding:32px;overflow:auto;color:var(--text-main); }
h1 { font-size:24px;margin:0 0 28px; }
section { padding:20px 0;border-top:1px solid var(--border-main); }
h2 { margin:0 0 12px;font-size:15px;color:var(--text-secondary); }
.more-entries { display:flex;flex-wrap:wrap;gap:12px 28px; }
</style>
