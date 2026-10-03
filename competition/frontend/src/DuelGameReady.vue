<script setup>
import { computed } from 'vue';
import { language } from './i18n.js';
import PlayerAvatar from './PlayerAvatar.vue';
defineProps({room:Object, busy:Boolean, projectName:Function, projectDescription:Function});
defineEmits(['ready']);
const en=computed(()=>language.value==='en');
</script>
<template>
  <section class="pregame-panel">
    <div class="pregame-heading"><p class="eyebrow">NEXT GAME</p><h1>{{ projectName(room.match.project_key) }}</h1><p>{{ en ? 'Both players must ready up. No automatic confirmation.' : '双方准备后开始，不会超时自动确认。' }}</p></div>
    <p>{{ $t(projectDescription(room.match.project_key)) }}</p>
    <div class="pregame-versus"><div v-for="side in ['yellow','white']" :key="side" :class="['pregame-team',side]">
      <h2 class="pregame-player"><PlayerAvatar :person="room.match.players[side]" />{{ room.match.players[side]?.display_name }}</h2>
      <strong>{{ room.match.readiness[side].player_ready ? (en ? 'Ready' : '已准备') : (en ? 'Not ready' : '未准备') }}</strong>
    </div></div>
    <p class="muted">{{ en ? 'If both players have not readied up, the room closes at' : '若双方仍未准备，房间将于' }} {{ new Date(room.waiting_expires_at).toLocaleString(en ? 'en-US' : 'zh-CN') }} {{ en ? '. Completed results are kept.' : '关闭，已完成的成绩保留。' }}</p>
    <div class="pregame-actions"><button v-if="room.me.can_mark_player_ready" class="primary-button" :disabled="busy" @click="$emit('ready','player')">{{ room.match.readiness[room.me.seat.side].player_ready ? (en ? 'Cancel ready' : '取消准备') : (en ? 'Ready for next game' : '准备下一局') }}</button></div>
  </section>
</template>
