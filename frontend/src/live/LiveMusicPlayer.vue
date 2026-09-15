<template>
  <section class="music-player" :class="{compact}">
    <header><Music2 :size="18" /><h2>{{ t('背景音乐', 'Background music') }}</h2>
      <span :title="t('列表循环', 'Repeat playlist')"><Repeat :size="16" /></span>
    </header>
    <select :value="selected" @change="selectTrack(Number($event.target.value), playing)" :aria-label="t('选择歌曲', 'Choose track')">
      <option v-for="(track, index) in tracks" :key="track.id" :value="index">{{ index + 1 }}. {{ track.title }}</option>
    </select>
    <div class="track-credit"><a v-if="current.source" :href="current.source" target="_blank" rel="noopener">{{ current.artist }} · CC0</a><span v-else>{{ t('本地歌单', 'Local playlist') }}</span></div>
    <div class="music-controls">
      <button @click="skip(-1)" :title="t('上一首', 'Previous track')"><SkipBack :size="18" /></button>
      <button @click="toggle" :title="playing ? t('暂停音乐', 'Pause music') : t('播放音乐', 'Play music')"><Pause v-if="playing" :size="19"/><Play v-else :size="19"/></button>
      <button @click="skip(1)" :title="t('下一首', 'Next track')"><SkipForward :size="18" /></button>
      <button @click="filesInput.click()" :title="t('添加本地音乐（可多选）', 'Add local music (multiple files)')"><FolderPlus :size="18" /></button>
      <button v-if="localTracks.length" @click="clearLocal" :title="t('清空本地歌单', 'Clear local tracks')"><Trash2 :size="17" /></button>
      <input ref="filesInput" type="file" accept="audio/*" multiple hidden @change="addFiles" />
      <input v-model="volume" type="range" min="0" max="1" step="0.05" :aria-label="t('音量', 'Volume')" />
    </div>
    <p v-if="error" role="status">{{ error }}</p>
    <audio ref="audio" preload="none" :volume="Number(volume)" @ended="selectTrack(nextTrackIndex(selected, tracks.length), true)" @error="failed" />
  </section>
</template>

<script setup>
import { computed, ref, onBeforeUnmount } from 'vue';
import { Music2, Repeat, SkipBack, SkipForward, Play, Pause, FolderPlus, Trash2 } from '@lucide/vue';
import { BUILTIN_TRACKS, nextTrackIndex, audioFileKey } from './playlist.js';
const props = defineProps({ lang: { type: String, default: 'zh' }, extraUrl: { type: String, default: '' },compact:Boolean });
const t = (zh, en) => props.lang === 'zh' ? zh : en;
const localTracks = ref([]), selected = ref(0), audio = ref(null), filesInput = ref(null);
const volume = ref(.25), playing = ref(false), activeSrc = ref(''), error = ref('');
let epoch = 0;
const tracks = computed(() => [...BUILTIN_TRACKS, ...localTracks.value, ...(props.extraUrl ? [{id:'configured',title:t('直播间选曲','Channel selection'),src:props.extraUrl}] : [])]);
const current = computed(() => tracks.value[selected.value] || tracks.value[0]);
function failed() {
  epoch++;
  playing.value = false;
  error.value = t('这首暂时无法播放，请切换其他歌曲。', 'This track is unavailable. Please choose another.');
}
async function selectTrack(index, autoplay = false) {
  const operation = ++epoch;
  audio.value.pause();
  playing.value = false;
  selected.value = nextTrackIndex(index, tracks.value.length, 0);
  error.value = '';
  // Do not fetch a selected track until the listener requests playback.
  activeSrc.value = '';
  audio.value.removeAttribute('src');
  if (!autoplay) { audio.value.load(); return; }
  activeSrc.value = current.value.src;
  audio.value.src = activeSrc.value;
  playing.value = true;
  try { await audio.value.play(); }
  catch { if (operation === epoch) failed(); }
}
function toggle() {
  if (playing.value) { epoch++; audio.value.pause(); playing.value = false; }
  else if (activeSrc.value === current.value.src) {
    const operation = ++epoch;
    playing.value = true;
    audio.value.play().catch(() => { if (operation === epoch) failed(); });
  } else selectTrack(selected.value, true);
}
function skip(delta) { selectTrack(nextTrackIndex(selected.value, tracks.value.length, delta), playing.value); }
function addFiles(event) {
  const known = new Set(localTracks.value.map(track => track.id));
  for (const file of Array.from(event.target.files || [])) {
    if (localTracks.value.length >= 100) break;
    const id = audioFileKey(file);
    if (known.has(id) || (!file.type.startsWith('audio/') && !/\.(mp3|m4a|ogg|wav|aac|flac|opus)$/i.test(file.name))) continue;
    localTracks.value.push({id,title:file.name,src:URL.createObjectURL(file)});
    known.add(id);
  }
  event.target.value = '';
}
function clearLocal() {
  if (selected.value >= BUILTIN_TRACKS.length) selectTrack(0, playing.value);
  for (const track of localTracks.value) URL.revokeObjectURL(track.src);
  localTracks.value = [];
}
onBeforeUnmount(() => {
  epoch++;
  audio.value?.pause();
  for (const track of localTracks.value) URL.revokeObjectURL(track.src);
});
</script>

<style scoped>
.music-player { border-top: 1px solid var(--border-main); padding-top: 18px; min-width: 0; }
header { display:flex; align-items:center; gap:8px; margin-bottom:12px; }
header h2 { font-size:15px; margin:0; }
header > span { margin-left:auto; color:var(--text-secondary); }
select { width:100%; min-width:0; padding:9px; color:var(--text-main); background:var(--bg-input); border:1px solid var(--border-main); border-radius:5px; font:inherit; color-scheme:light; }
:global(:root[data-theme='dark'] .music-player select) { color-scheme:dark; --live-option-background:#1e293b; }
option { color:var(--text-main); background-color:var(--live-option-background, #fffcf6); }
.track-credit { font-size:11px; margin:8px 0 12px; color:var(--text-secondary); overflow-wrap:anywhere; }
.track-credit a { color:inherit; }
.music-controls { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
button { display:inline-flex; align-items:center; justify-content:center; width:36px; height:36px; border:1px solid var(--border-main); border-radius:6px; background:var(--bg-card); color:var(--text-main); cursor:pointer; }
button:hover { border-color:var(--accent); }
input[type=range] { min-width:60px; width:80px; flex:1; accent-color:var(--accent); }
p { font-size:12px; color:var(--text-secondary); }
.compact { display:grid;grid-template-columns:auto minmax(180px,1fr) auto auto;gap:14px;align-items:center;padding:10px 0; }
.compact header { margin:0; }.compact h2 { font-size:13px; }.compact select { padding:6px;font-size:12px;height:32px; }.compact .music-controls { grid-column:3;grid-row:1; }.compact button { width:30px;height:30px;border-radius:4px; }.compact .track-credit { grid-column:4;grid-row:1;margin:0;font-size:10px; }.compact p { grid-column:1 / -1;margin:0; }
</style>
