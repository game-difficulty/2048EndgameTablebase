<script setup>
import { computed, ref } from 'vue';
import { language } from './i18n.js';
import PlayerAvatar from './PlayerAvatar.vue';
import PublicPredictionWindow from './PublicPredictionWindow.vue';
import { publicRoomMode, modeText } from './publicRoomModes.js';
const props = defineProps({room: Object, busy: Boolean, projectName: Function});
const mode = computed(()=>publicRoomMode(props.room.room_kind));
const emit = defineEmits(['claim','leave','ready']);
const en = computed(() => language.value==='en');
const copied = ref(false);
async function copy() { try { await navigator.clipboard.writeText(window.location.href); copied.value=true; } catch { copied.value=false; } }
</script>
<template>
  <section class="panel duel-waiting">
    <div class="duel-wait-heading"><div><p class="eyebrow">{{modeText(mode?.title,language)}} · 1 VS 1</p><h2>{{ en ? 'Waiting for both players' : '等待双方准备' }}</h2></div><button class="secondary-button" @click="copy">{{ copied ? (en ? 'Copied' : '已复制') : (en ? 'Copy invite link' : '复制邀请链接') }}</button></div>
    <PublicPredictionWindow :room="room" />
    <p>{{ room.rules.predictions_enabled ? (en?'Both players must ready up, followed by a 60-second spectator betting window.':'双方主动准备后，先进入 60 秒观众下注窗口。') : (en ? 'The first game starts when both players are ready. There is no automatic ready-up.' : '双方准备后直接开始第一局，不会超时自动准备。') }}</p>
    <div class="duel-players"><article v-for="side in ['yellow','white']" :key="side" :class="side">
      <PlayerAvatar v-if="room.seats.find(s=>s.side===side)" :person="room.seats.find(s=>s.side===side)" />
      <h3>{{ room.seats.find(s=>s.side===side)?.display_name || (en ? 'Open seat' : '等待对手') }}</h3>
      <strong>{{ room.teams[side].ready ? (en ? 'Ready' : '已准备') : (en ? 'Not ready' : '未准备') }}</strong>
      <button v-if="!room.seats.some(s=>s.side===side) && !room.me.seat && room.me.can_claim_seat" class="primary-button" :disabled="busy" @click="emit('claim',side,1)">{{ en ? 'Take this seat' : '加入对决' }}</button>
    </article></div>
    <h3 v-if="mode?.showProjectOrder">{{ en ? 'Project order' : '项目顺序' }}</h3>
    <ol v-if="mode?.showProjectOrder"><li v-for="key in room.rules.game_keys" :key="key">{{ projectName(room.selected_projects[key]) }}</li></ol>
    <p class="muted">{{ modeText(mode?.waitingClock,language) }} {{ room.rules.team_clock_seconds/60 }} {{ en ? 'minutes' : '分钟' }}</p>
    <p class="muted">{{ en ? 'If the match has not started, this room closes at' : '若仍未开赛，房间将于' }} {{ new Date(room.waiting_expires_at).toLocaleString(en ? 'en-US' : 'zh-CN') }} {{ en ? '' : '关闭。' }}</p>
    <div class="action-buttons"><button v-if="room.me.can_leave_seat" class="secondary-button" :disabled="busy" @click="emit('leave')">{{ en ? 'Leave seat' : '离开席位' }}</button><button v-if="room.me.can_ready" class="primary-button" :disabled="busy" @click="emit('ready')">{{ room.teams[room.me.seat.side].ready ? (en ? 'Cancel ready' : '取消准备') : (en ? 'Ready' : '准备') }}</button></div>
  </section>
</template>
<style scoped>
.duel-waiting{padding:24px}@media(max-width:600px){.duel-waiting{padding:16px}}
.duel-wait-heading{display:flex;align-items:center;justify-content:space-between;gap:12px}.duel-waiting p{line-height:1.7}.duel-players{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:24px 0}.duel-players article{border:1px solid #8884;border-radius:16px;padding:24px;display:flex;flex-direction:column;gap:12px;align-items:center}.duel-players h3{overflow-wrap:anywhere;text-align:center;margin:0}.duel-waiting ol{display:flex;gap:12px 32px;flex-wrap:wrap;padding-left:24px}.duel-waiting .action-buttons{justify-content:flex-end}@media(max-width:600px){.duel-wait-heading{align-items:start;flex-direction:column}.duel-players{gap:10px}.duel-players article{padding:16px 8px}}
</style>
