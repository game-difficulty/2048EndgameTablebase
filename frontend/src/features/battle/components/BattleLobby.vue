<template>
  <div class="battle-lobby">
    <header class="battle-lobby-banner">
      <div>
        <span class="ui-caption font-black uppercase text-text-secondary">{{ $t('battle.lobby.kicker') }}</span>
        <div class="battle-room-code-line">
          <h2>{{ room.room_code }}</h2>
          <button type="button" class="battle-copy-btn" @click="copyInvite">{{ copied ? $t('common.copied') : $t('battle.actions.copyInvite') }}</button>
        </div>
      </div>
      <div class="battle-lobby-summary">
        <strong>{{ room.full_pattern }}</strong>
        <span>{{ $t(`battle.status.${room.status}`) }}</span>
        <time v-if="room.expires_at">{{ remainingLabel }}</time>
      </div>
    </header>

    <div class="battle-lobby-grid">
      <section class="battle-seats-panel">
        <div class="battle-panel-title">
          <div><h3>{{ $t('battle.lobby.players') }}</h3><p>{{ players.length }}/{{ room.max_players }}</p></div>
          <div v-if="selfMember && ['preparing', 'waiting'].includes(room.status)" class="battle-role-segment">
            <button type="button" :class="selfMember.role === 'player' ? 'active' : ''" :disabled="players.length >= room.max_players && selfMember.role !== 'player'" @click="$emit('role', 'player')">{{ $t('battle.roles.player') }}</button>
            <button type="button" :class="selfMember.role === 'spectator' ? 'active' : ''" :disabled="!room.allow_spectators" @click="$emit('role', 'spectator')">{{ $t('battle.roles.spectator') }}</button>
          </div>
        </div>

        <div class="battle-seat-grid">
          <article v-for="seat in seatRows" :key="seat.index" :class="['battle-seat', seat.member ? 'occupied' : 'empty']">
            <template v-if="seat.member">
              <div class="battle-seat-avatar">
                <img v-if="seat.member.avatar_url" :src="seat.member.avatar_url" alt="" />
                <span v-else>{{ initials(seat.member.display_name) }}</span>
                <i :class="seat.member.online ? 'online' : 'offline'" aria-hidden="true"></i>
              </div>
              <div class="battle-seat-identity">
                <strong>{{ seat.member.display_name }}</strong>
                <span>{{ seat.member.user_id === room.host_user_id ? $t('battle.roles.host') : (seat.member.ready ? $t('battle.status.ready') : $t('battle.status.not_ready')) }}</span>
              </div>
              <button v-if="isHost && seat.member.user_id !== selfMember?.user_id && ['preparing', 'waiting'].includes(room.status)" type="button" class="battle-kick-btn" :title="$t('battle.actions.kick')" @click="$emit('kick', seat.member.user_id)">×</button>
            </template>
            <template v-else>
              <span class="battle-empty-seat-number">{{ String(seat.index + 1).padStart(2, '0') }}</span>
              <span>{{ $t('battle.lobby.openSeat') }}</span>
            </template>
          </article>
        </div>

        <div class="battle-lobby-actions">
          <button v-if="selfMember?.role === 'player'" type="button" :class="['battle-ready-btn', selfMember.ready ? 'ready' : '']" :disabled="room.status !== 'waiting'" @click="$emit('ready', !selfMember.ready)">
            {{ selfMember.ready ? $t('battle.actions.cancelReady') : $t('battle.actions.ready') }}
          </button>
          <button v-if="isHost" type="button" class="battle-start-btn" :disabled="!canStart" @click="$emit('start')">{{ $t('battle.actions.start') }}</button>
          <button type="button" class="battle-leave-btn" @click="$emit('leave')">{{ isHost ? $t('battle.actions.closeRoom') : $t('battle.actions.leave') }}</button>
        </div>
        <p class="battle-start-rule">{{ $t(room.allow_spectators ? 'battle.lobby.startRuleSpectators' : 'battle.lobby.startRuleNoSpectators') }}</p>
      </section>

      <aside class="battle-room-sidebar">
        <section class="battle-settings-summary">
          <div class="battle-panel-title"><div><h3>{{ $t('battle.lobby.settings') }}</h3><p>{{ $t('battle.lobby.settingsLocked') }}</p></div></div>
          <slot name="settings-summary" :room="room">
            <dl>
              <div><dt>{{ $t('battle.form.pattern') }}</dt><dd>{{ room.full_pattern }}</dd></div>
              <div><dt>{{ $t('battle.form.initialBoard') }}</dt><dd class="font-mono">{{ displayedInitialBoard }}</dd></div>
              <div><dt>{{ $t(stepsLabelKey) }}</dt><dd>{{ room.max_steps || $t('battle.form.unlimited') }}</dd></div>
              <div><dt>{{ $t('battle.form.stepTimeout') }}</dt><dd>{{ room.step_timeout_seconds }}s</dd></div>
              <div><dt>{{ $t('battle.form.publicRoom') }}</dt><dd>{{ room.visibility === 'public' ? $t('common.yes') : $t('common.no') }}</dd></div>
              <div><dt>{{ $t('battle.form.allowSpectators') }}</dt><dd>{{ room.allow_spectators ? $t('common.yes') : $t('common.no') }}</dd></div>
              <div><dt>{{ $t('battle.form.chatRoles') }}</dt><dd>{{ chatRolesLabel }}</dd></div>
            </dl>
          </slot>
        </section>
        <section class="battle-spectator-list">
          <div class="battle-panel-title"><div><h3>{{ $t('battle.lobby.spectators') }}</h3><p>{{ spectators.length }}</p></div></div>
          <div v-if="!spectators.length" class="battle-sidebar-empty">{{ $t('battle.lobby.noSpectators') }}</div>
          <div v-else class="battle-spectator-chips">
            <span v-for="member in spectators" :key="member.user_id">{{ member.display_name }}</span>
          </div>
        </section>
      </aside>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps({
  room: { type: Object, required: true },
  members: { type: Array, default: () => [] },
  currentUserId: { type: Number, default: 0 },
  now: { type: Number, default: () => Date.now() },
});

defineEmits(['ready', 'start', 'kick', 'leave', 'role']);
const { t } = useI18n();
const copied = ref(false);
const players = computed(() => props.members.filter((member) => member.role === 'player' && member.status === 'active'));
const spectators = computed(() => props.members.filter((member) => member.role === 'spectator' && member.status === 'active'));
const selfMember = computed(() => props.members.find((member) => Number(member.user_id) === Number(props.currentUserId)) || null);
const isHost = computed(() => Number(props.room.host_user_id) === Number(props.currentUserId));
const readyPlayers = computed(() => players.value.filter((member) => member.ready));
const canStart = computed(() => (
  isHost.value
  && Boolean(selfMember.value?.ready)
  && readyPlayers.value.length >= 2
  && props.room.status === 'waiting'
));
const seatRows = computed(() => Array.from({ length: Number(props.room.max_players || 2) }, (_unused, index) => ({ index, member: players.value[index] || null })));
const displayedInitialBoard = computed(() => (
  props.room.route?.initial_board
  || props.room.initial_board
  || t('battle.status.preparingInitialBoard')
));
const stepsLabelKey = computed(() => (
  props.room.mode_settings?.score_step_limit
    ? 'battle.form.scoredSteps'
    : 'battle.form.maxSteps'
));
const chatRolesLabel = computed(() => {
  const roles = Array.isArray(props.room.chat_roles)
    ? props.room.chat_roles
    : ['host', 'player', 'spectator'];
  return roles.length
    ? roles.map((role) => t(`battle.roles.${role}`)).join(' / ')
    : t('battle.form.chatRolesNone');
});
const remainingLabel = computed(() => {
  const remaining = Math.max(0, Date.parse(props.room.expires_at || '') - Number(props.now));
  const minutes = Math.floor(remaining / 60000);
  const seconds = Math.floor((remaining % 60000) / 1000);
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
});
const initials = (name) => String(name || '?').trim().slice(0, 2).toUpperCase();
const copyInvite = async () => {
  const url = `${window.location.origin}${window.location.pathname}?tab=battle&room=${props.room.room_code}`;
  const invite = t('battle.lobby.inviteText', {
    tablebase: props.room.full_pattern,
    url,
  });
  await navigator.clipboard?.writeText?.(invite);
  copied.value = true;
  window.setTimeout(() => { copied.value = false; }, 1600);
};
</script>

<style scoped>
.battle-lobby { display: flex; flex-direction: column; gap: 16px; }
.battle-lobby-banner { min-height: 92px; display: flex; align-items: center; justify-content: space-between; gap: 20px; padding: 17px 20px; border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-card); box-shadow: 0 14px 32px rgba(0,0,0,.07); }
.battle-room-code-line { display: flex; align-items: center; gap: 12px; margin-top: 4px; }
.battle-room-code-line h2 { margin: 0; color: var(--text-main); font: 900 28px/1 var(--font-mono, monospace); letter-spacing: .08em; }
.battle-copy-btn, .battle-kick-btn { border: 1px solid var(--border-main); border-radius: 7px; background: var(--bg-main); color: var(--text-main); font-size: var(--font-ui-xs); font-weight: 900; padding: 6px 10px; }
.battle-lobby-summary { display: grid; grid-template-columns: auto auto auto; align-items: center; gap: 12px; }
.battle-lobby-summary strong, .battle-lobby-summary span, .battle-lobby-summary time { border-left: 1px solid var(--border-main); padding-left: 12px; color: var(--text-main); font-size: var(--font-ui-sm); }
.battle-lobby-summary span { color: var(--accent); font-weight: 900; }
.battle-lobby-summary time { font-family: var(--font-mono, monospace); font-weight: 900; }
.battle-lobby-grid { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(330px, .8fr); gap: 16px; }
.battle-seats-panel, .battle-settings-summary, .battle-spectator-list { border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-card); box-shadow: 0 14px 32px rgba(0,0,0,.06); }
.battle-seats-panel { padding: 18px; }
.battle-room-sidebar { display: flex; flex-direction: column; gap: 16px; }
.battle-settings-summary, .battle-spectator-list { padding: 17px; }
.battle-panel-title { display: flex; align-items: center; justify-content: space-between; min-height: 40px; margin-bottom: 13px; }
.battle-panel-title > div:first-child { display: flex; align-items: baseline; gap: 9px; }
.battle-panel-title h3 { margin: 0; color: var(--text-main); font-size: var(--font-ui-base); font-weight: 900; }
.battle-panel-title p { margin: 0; color: var(--text-secondary); font-size: var(--font-ui-xs); }
.battle-role-segment { display: flex; border: 1px solid var(--border-main); border-radius: 7px; overflow: hidden; }
.battle-role-segment button { min-width: 80px; padding: 7px 10px; border: 0; background: var(--bg-main); color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 900; }
.battle-role-segment button.active { background: var(--btn-bg); color: white; }
.battle-role-segment button:disabled { opacity: .4; }
.battle-seat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
.battle-seat { min-height: 68px; display: flex; align-items: center; gap: 11px; padding: 10px 12px; border: 1px solid var(--border-main); border-radius: 7px; position: relative; }
.battle-seat.occupied { background: color-mix(in srgb, var(--bg-main) 72%, transparent); }
.battle-seat.empty { justify-content: center; color: var(--text-secondary); border-style: dashed; font-size: var(--font-ui-xs); }
.battle-empty-seat-number { color: var(--accent); font-family: var(--font-mono, monospace); font-weight: 900; }
.battle-seat-avatar { width: 42px; height: 42px; position: relative; flex: 0 0 auto; }
.battle-seat-avatar img, .battle-seat-avatar span { width: 100%; height: 100%; border-radius: 50%; }
.battle-seat-avatar img { object-fit: cover; }
.battle-seat-avatar span { display: grid; place-items: center; border: 1px solid var(--border-main); color: var(--accent); font-size: 12px; font-weight: 900; }
.battle-seat-avatar i { position: absolute; right: 0; bottom: 0; width: 9px; height: 9px; border-radius: 50%; border: 2px solid var(--bg-card); background: #8993a1; }
.battle-seat-avatar i.online { background: #35a96b; }
.battle-seat-identity { min-width: 0; display: flex; flex: 1; flex-direction: column; gap: 4px; }
.battle-seat-identity strong { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-main); font-size: var(--font-ui-sm); }
.battle-seat-identity span { color: var(--text-secondary); font-size: var(--font-ui-xs); }
.battle-kick-btn { width: 28px; height: 28px; padding: 0; color: #dc4c4c; }
.battle-lobby-actions { display: grid; grid-template-columns: 1fr 1fr auto; gap: 9px; margin-top: 15px; }
.battle-ready-btn, .battle-start-btn, .battle-leave-btn { min-height: 42px; border: 1px solid var(--border-main); border-radius: 7px; font-size: var(--font-ui-sm); font-weight: 900; }
.battle-ready-btn { background: var(--bg-main); color: var(--text-main); }
.battle-ready-btn.ready { border-color: #3ba86b; color: #258453; }
.battle-start-rule { margin: 8px 0 0; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 700; }
.battle-start-btn { background: var(--btn-bg); border-color: var(--btn-bg); color: white; }
.battle-start-btn:disabled { opacity: .38; cursor: not-allowed; }
.battle-leave-btn { padding: 0 18px; background: transparent; color: var(--text-secondary); }
.battle-settings-summary dl { margin: 0; }
.battle-settings-summary dl div { display: grid; grid-template-columns: minmax(105px,.75fr) minmax(0,1.25fr); gap: 10px; padding: 9px 0; border-top: 1px solid color-mix(in srgb, var(--border-main) 70%, transparent); }
.battle-settings-summary dt { color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 800; }
.battle-settings-summary dd { margin: 0; overflow: hidden; text-overflow: ellipsis; color: var(--text-main); font-size: var(--font-ui-xs); font-weight: 900; text-align: right; }
.battle-sidebar-empty { padding: 22px 0; color: var(--text-secondary); text-align: center; font-size: var(--font-ui-xs); }
.battle-spectator-chips { display: flex; flex-wrap: wrap; gap: 7px; }
.battle-spectator-chips span { max-width: 100%; overflow: hidden; text-overflow: ellipsis; border: 1px solid var(--border-main); border-radius: 999px; padding: 5px 9px; color: var(--text-main); font-size: var(--font-ui-xs); }
</style>
