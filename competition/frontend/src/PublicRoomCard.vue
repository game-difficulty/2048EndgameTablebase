<script setup>
import { computed } from 'vue';
import { language } from './i18n.js';
import { publicRoomMode, publicRoomStatus, modeText } from './publicRoomModes.js';
const props=defineProps({room:Object,busy:Boolean});
const emit=defineEmits(['navigate','close']);
const en=computed(()=>language.value==='en');
const mode=computed(()=>publicRoomMode(props.room.room_kind));
</script>
<template>
  <article class="public-room-card">
    <button type="button" class="public-room-open" @click="emit('navigate',`/rooms/${room.room_code}`)">
      <span class="room-card-copy"><small>{{modeText(mode?.title,language)}} · {{room.room_code}}</small><strong>{{room.name}}</strong><span>{{room.occupied_seats ?? 0}} / 2 {{en?'players':'人'}} · {{room.rules?.team_clock_seconds/60}} {{en?'min':'分钟'}}<template v-if="mode?.showProjectOrder"> · {{room.rules?.game_count}} {{en?'projects':'个项目'}}</template></span></span>
      <span class="room-card-state"><b>{{publicRoomStatus(room.status,language)}}</b><small>{{en?'Open room →':'进入房间 →'}}</small></span>
    </button>
    <button v-if="room.can_close" type="button" class="room-card-close" :disabled="busy" :aria-label="`${en?'Close room':'关闭房间'} ${room.name}`" @click="emit('close',room)">{{en?'Close':'关闭'}}</button>
  </article>
</template>
<style scoped>
.public-room-card{display:flex;align-items:center;border:1px solid var(--competition-border);border-radius:12px;background:var(--competition-card);overflow:hidden}.public-room-open{display:flex;align-items:center;justify-content:space-between;flex:1;min-width:0;gap:18px;padding:18px;background:transparent;border:0;color:inherit;text-align:left;font:inherit;cursor:pointer}.public-room-open:hover{background:#b88b3709}.room-card-copy{display:grid;gap:7px;min-width:0}.room-card-copy strong{font-size:17px;overflow-wrap:anywhere}.room-card-copy small,.room-card-copy>span{font-size:12px;color:var(--competition-muted)}.room-card-state{display:grid;gap:8px;flex-shrink:0;text-align:right;font-size:13px}.room-card-state small{font-size:12px;color:var(--competition-muted)}.room-card-close{background:transparent;border:0;border-left:1px solid var(--competition-border);color:var(--competition-muted);padding:12px;min-height:44px;cursor:pointer}@media(max-width:600px){.public-room-open{padding:14px;align-items:start;gap:10px}.room-card-copy strong{font-size:16px}.room-card-state{font-size:12px}.room-card-close{padding:10px}}
</style>
