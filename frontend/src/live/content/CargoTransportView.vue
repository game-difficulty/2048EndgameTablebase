<template>
  <div class="cargo-live-shell">
    <div class="cargo-live-header"><span><small>{{ metric.label }}</small><strong>{{ metric.value }}</strong></span><span><small>{{ t('用时', 'TIME') }}</small><strong>{{ countdown }}</strong></span></div>
    <div class="cargo-live-stage">
      <i v-for="index in 16" :key="`cell-${index}`" class="grid-cell" :style="position(Math.floor((index - 1) / 4), (index - 1) % 4)" />
      <b v-for="(value, index) in cells" v-show="value > 0 && !hidden.has(index)" :key="`tile-${index}-${value}`"
        :class="['number-tile', { pop: pops.has(index), appear: appear === index }]"
        :style="{ ...position(Math.floor(index / 4), index % 4), ...tileColor(value) }">{{ value }}</b>
      <b v-for="(tile, index) in moving" :key="`moving-${index}`" class="number-tile moving" :style="movingStyle(tile)">{{ tile.value }}</b>
      <div class="live-port" aria-hidden="true">
        <i v-for="index in 4" :key="`entry-${index}`" :style="position(-2 + Math.floor((index - 1) / 2), 1 + (index - 1) % 2)" />
        <i v-for="col in [1, 2]" :key="`exit-${col}`" :style="position(4, col)" />
        <span class="entry-label">{{ t('入口', 'IN') }}</span><span class="exit-label">{{ t('出口 ↓', 'OUT ↓') }}</span>
      </div>
      <div v-if="visualCargo" :key="visualCargo.id" :class="['special-cargo', { instant: cargoInstant, 'cao-cargo': visualCargo.shape === 0 }]" :style="cargoPosition(visualCargo)">
        <span v-for="(cell, index) in shapeCells(visualCargo)" :key="index" class="cargo-cell" :style="partPosition(cell)" />
        <span v-for="(bridge, index) in bridges(visualCargo)" :key="`bridge-${index}`" class="cargo-bridge" :style="bridge" />
        <strong :style="cargoLabelStyle(visualCargo)">{{ visualCargo.shape === 0 ? '曹' : '◆' }}</strong>
      </div>
    </div>
    <small>{{ payload.move_count || 0 }} {{ t('步', 'moves') }}</small>
  </div>
</template>

<script setup>
import { shouldRefreshLiveClock } from '../displayClock.js';
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { projectPerformanceMetric } from '../../../../competition/shared/projectMetrics.mjs';
import { CARGO_SHAPES } from '../../../../competition/shared/cargoShapes.mjs';
import { liveTileColors } from '../tilePalette.js';

const props = defineProps({ view: { type: Object, required: true }, lang: { type: String, default: 'zh' }, suspended: Boolean });
const payload = computed(() => props.view?.payload || {});
const metric = computed(() => projectPerformanceMetric(props.view, props.lang));
const cells = computed(() => (payload.value.board || []).flat());
const visualCargo = ref(null), cargoInstant = ref(true);
const hidden = ref(new Set()), pops = ref(new Set()), appear = ref(null), moving = ref([]), started = ref(false);
const now = ref(performance.now()), receivedAt = ref(performance.now()), elapsedAnchor = ref(0);
let timer = null, timers = [], epoch = 0;
const GAP = 2.25, CELL = (100 - GAP * 5) / 4, PITCH = CELL + GAP, BOX = CELL * 2 + GAP;
const t = (zh, en) => props.lang === 'zh' ? zh : en;
const tileColor = value => liveTileColors(value);
const position = (row, col) => ({
  left: `${GAP + col * PITCH}%`, top: `${(50 + GAP + row * PITCH) / 175 * 100}%`,
  width: `${CELL}%`, height: `${CELL / 175 * 100}%`,
});
const cargoPosition = cargo => ({
  left: `${GAP + cargo.col * PITCH}%`, top: `${(50 + GAP + cargo.row * PITCH) / 175 * 100}%`,
  width: `${BOX}%`, height: `${BOX / 175 * 100}%`,
});
const shapeCells = cargo => CARGO_SHAPES[cargo.shape]?.cells || [];
function cargoLabelStyle(cargo) {
  if (cargo.shape === 0) return null;
  const cells = shapeCells(cargo);
  const joint = cells.find(([row, col]) => cells.filter(([otherRow, otherCol]) =>
    Math.abs(row - otherRow) + Math.abs(col - otherCol) === 1).length === 2);
  if (!cells.length) return null;
  const center = joint || [
    cells.reduce((sum, cell) => sum + cell[0], 0) / cells.length,
    cells.reduce((sum, cell) => sum + cell[1], 0) / cells.length,
  ];
  return {
    left: `${(center[1] * PITCH + CELL / 2) / BOX * 100}%`,
    top: `${(center[0] * PITCH + CELL / 2) / BOX * 100}%`,
  };
}
const partPosition = ([row, col]) => ({
  left: `${col * PITCH / BOX * 100}%`, top: `${row * PITCH / BOX * 100}%`,
  width: `${CELL / BOX * 100}%`, height: `${CELL / BOX * 100}%`,
});
function bridges(cargo) {
  const occupied = new Set(shapeCells(cargo).map(([row, col]) => `${row},${col}`));
  const result = [];
  for (const [row, col] of shapeCells(cargo)) {
    if (occupied.has(`${row},${col + 1}`)) result.push({ left: `calc(${CELL / BOX * 100}% - 1px)`, top: `${row * PITCH / BOX * 100}%`, width: `calc(${GAP / BOX * 100}% + 2px)`, height: `${CELL / BOX * 100}%` });
    if (occupied.has(`${row + 1},${col}`)) result.push({ left: `${col * PITCH / BOX * 100}%`, top: `calc(${CELL / BOX * 100}% - 1px)`, width: `${CELL / BOX * 100}%`, height: `calc(${GAP / BOX * 100}% + 2px)` });
  }
  if (occupied.size === 4) result.push({ left: `${CELL / BOX * 100}%`, top: `${CELL / BOX * 100}%`, width: `${GAP / BOX * 100}%`, height: `${GAP / BOX * 100}%` });
  return result;
}
const countdown = computed(() => {
  const elapsed = elapsedAnchor.value + (payload.value.finished || props.suspended ? 0 : Math.max(0, now.value - receivedAt.value));
  const ms = Math.max(0, elapsed);
  return `${String(Math.floor(ms / 60000)).padStart(2, '0')}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')}.${String(Math.floor(ms / 10) % 100).padStart(2, '0')}`;
});
const movingStyle = item => ({
  ...position(Math.floor((started.value ? item.to : item.from) / 4), (started.value ? item.to : item.from) % 4),
  ...tileColor(item.value), transition: 'left 100ms ease-in-out, top 100ms ease-in-out',
});
function clearTimers() { timers.forEach(clearTimeout); timers = []; }

watch(() => props.view?.sequence, async () => {
  const currentEpoch = ++epoch;
  clearTimers();
  elapsedAnchor.value = Number(payload.value.elapsed_ms || 0);
  receivedAt.value = performance.now();
  now.value = receivedAt.value;
  const transition = payload.value.last_transition;
  hidden.value = new Set(); pops.value = new Set(); appear.value = null; moving.value = []; started.value = false;
  if (transition?.kind !== 'move') {
    visualCargo.value = payload.value.cargo || null;
    cargoInstant.value = true;
    return;
  }
  const destinations = (transition.movements || []).map(item => item.to);
  if (transition.spawn?.index != null) destinations.push(transition.spawn.index);
  hidden.value = new Set(destinations);
  moving.value = (transition.movements || []).map(item => ({ ...item }));
  visualCargo.value = transition.cargoMoved ? transition.cargoBefore : payload.value.cargo;
  cargoInstant.value = true;
  await nextTick();
  if (currentEpoch !== epoch) return;
  void document.body.offsetHeight;
  requestAnimationFrame(() => {
    if (currentEpoch !== epoch) return;
    started.value = true;
    if (transition.cargoMoved) {
      cargoInstant.value = false;
      visualCargo.value = transition.cargoExit || payload.value.cargo;
    }
  });
  timers.push(setTimeout(() => {
    if (currentEpoch !== epoch) return;
    moving.value = [];
    hidden.value = new Set(transition.spawn?.index == null ? [] : [transition.spawn.index]);
    pops.value = new Set((transition.movements || []).filter(item => item.merged).map(item => item.to));
    if (transition.cargoExit) { visualCargo.value = payload.value.cargo; cargoInstant.value = true; }
  }, 100));
  timers.push(setTimeout(() => {
    if (currentEpoch !== epoch) return;
    hidden.value = new Set(); appear.value = transition.spawn?.index ?? null;
  }, 125));
  timers.push(setTimeout(() => { if (currentEpoch === epoch) { pops.value = new Set(); appear.value = null; } }, 300));
}, { immediate: true });

watch(() => [payload.value.elapsed_ms, props.suspended], ([elapsed]) => {
  elapsedAnchor.value = Number(elapsed || 0);
  receivedAt.value = performance.now();
  now.value = receivedAt.value;
});

onMounted(() => { timer = setInterval(() => { if (!payload.value.finished && !props.suspended && shouldRefreshLiveClock()) now.value = performance.now(); }, 50); });
onBeforeUnmount(() => { ++epoch; clearTimers(); clearInterval(timer); });
</script>

<style scoped>
.cargo-live-shell{height:100%;display:grid;grid-template-rows:48px minmax(0,1fr) 20px;gap:5px;align-items:center;container-type:size}
.cargo-live-header{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));align-items:end;gap:8px;border-bottom:1px solid #334155;padding-bottom:5px}.cargo-live-header span{display:grid;gap:3px;min-width:0}.cargo-live-header span:last-child{text-align:right}.cargo-live-header small{color:#94a3b8;font-size:10px;letter-spacing:.08em}.cargo-live-header strong{overflow:hidden;color:#f8fafc;font-size:24px;font-variant-numeric:tabular-nums;line-height:1;text-overflow:ellipsis;white-space:nowrap}
.cargo-live-stage{position:relative;width:min(100cqw,245px,calc((100cqh - 78px)*4/7));aspect-ratio:4/7;margin:auto;overflow:hidden}.cargo-live-stage::before{content:"";position:absolute;left:0;top:28.5714285714%;width:100%;height:57.1428571429%;border-radius:7px;background:#25334b}.grid-cell,.number-tile,.live-port i{position:absolute;box-sizing:border-box;border-radius:5px}.grid-cell{background:#3b4960}.number-tile{z-index:2;display:grid;place-items:center;font-size:clamp(13px,2vw,27px);font-weight:800}.number-tile.moving{z-index:3}.number-tile.pop{animation:pop .2s ease backwards}.number-tile.appear{animation:appear .2s ease backwards}
/* Keep guides below cargo, but labels above it; the port must not create a stacking context. */
.live-port{position:absolute;inset:0;pointer-events:none}.live-port i{z-index:1;border:1px dashed #9d8658}.live-port span{position:absolute;z-index:5;left:50%;transform:translateX(-50%);color:#d8bd69;font-size:9px;white-space:nowrap}.entry-label{top:0}.exit-label{bottom:0}
.special-cargo{position:absolute;z-index:4;background:transparent;transition:left .1s ease-in-out,top .1s ease-in-out}.special-cargo.instant{transition:none}.cargo-cell,.cargo-bridge{position:absolute;background:#448d8a}.cargo-cell{border-radius:5px}.special-cargo strong{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);color:#fff;font-size:25px;text-shadow:0 1px 2px #234}.cargo-live-shell>small{text-align:center;color:#94a3b8}
@keyframes pop{50%{transform:scale(1.2)}}@keyframes appear{from{transform:scale(0);opacity:0}}
.special-cargo.cao-cargo{background:#d8ab5d;border-radius:5px}
.cao-cargo .cargo-cell,.cao-cargo .cargo-bridge{visibility:hidden}
.special-cargo.cao-cargo{container-type:inline-size}
.special-cargo.cao-cargo strong{color:#50371b;font-size:60cqw;font-weight:800;line-height:1;text-shadow:none}
</style>
