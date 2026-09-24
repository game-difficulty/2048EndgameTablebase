<template>
  <AuxiliaryEntry :entry="AUXILIARY_ENTRIES.live" :live="online" class="top-live-entry" />
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';
import AuxiliaryEntry from './AuxiliaryEntry.vue';
import { AUXILIARY_ENTRIES } from '../app/auxiliaryEntries.js';
import { createLiveStatusPoller } from '../services/live/liveStatus.js';
import { getBackendUrl } from '../services/runtime/backendUrl.js';

const online = ref(false);
let stop;
onMounted(() => {
  stop = createLiveStatusPoller({
    url: getBackendUrl('/api/live/status'),
    onChange: value => { online.value = value; },
  });
});
onUnmounted(() => stop?.());
</script>
