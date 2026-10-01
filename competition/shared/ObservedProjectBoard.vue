<template>
  <CargoBoard v-if="shown?.view_kind === 'cargo-transport'" :data-frame-sequence="shown.sequence" :snapshot="snapshot" :tile-styles="tileStyles" :font-scale="fontScale" disabled />
  <PolyominoBoard v-else-if="shown?.view_kind === 'polyomino-board'" :data-frame-sequence="shown.sequence" :snapshot="snapshot" :tile-styles="tileStyles" :font-scale="fontScale" disabled />
  <TournamentBoard v-else-if="shown" :data-frame-sequence="shown.sequence" :snapshot="snapshot" :tile-styles="tileStyles" :font-scale="fontScale" disabled
    :mirror-portals="Boolean(payload.mirror_portals)" :irregular-shape="Boolean(payload.shape_shifter)"
    :aftershock="Boolean(payload.aftershock)" :sealed-cells="payload.sealed_cells || []" />
</template>

<script setup>
import { computed, onBeforeUnmount, shallowRef, watch } from 'vue';
import CargoBoard from '../frontend/src/projects/CargoBoard.vue';
import PolyominoBoard from '../frontend/src/projects/PolyominoBoard.vue';
import TournamentBoard from '../frontend/src/projects/TournamentBoard.vue';
import { ProjectPlayback } from './projectPlayback.mjs';

const props = defineProps({ view: Object, streamKey: String, tileStyles: Object, fontScale: { type: Number, default: 1 } });
const emit = defineEmits(['present', 'pending', 'gap']);
const shown = shallowRef(null);
const playback = new ProjectPlayback(view => { shown.value = view; emit('present', view); }, {
  onPending: value => emit('pending', value), onGap: value => emit('gap', value),
});
const payload = computed(() => shown.value?.payload || {});
const snapshot = computed(() => ({ ...payload.value,
  board: (payload.value.board || []).flat(), revision: Number(shown.value?.sequence || 0),
  transition: payload.value.last_transition || null,
}));
watch(() => [props.view, props.streamKey], () => playback.receive(props.view, props.streamKey), { immediate: true });
onBeforeUnmount(() => playback.close());
</script>
