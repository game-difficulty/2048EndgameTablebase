<template>
  <div class="battle-lobby">
    <header class="battle-lobby-banner">
      <div>
        <span class="ui-caption font-black uppercase text-text-secondary">{{ $t('battle.lobby.kicker') }}</span>
        <div class="battle-room-code-line">
          <h2>{{ room.room_code }}</h2>
          <span v-if="permanentRoom" class="battle-lobby-permanent">{{ $t('battle.room.permanent') }}</span>
          <button type="button" class="battle-copy-btn" @click="copyInvite">{{ copied ? $t('common.copied') : $t('battle.actions.copyInvite') }}</button>
        </div>
      </div>
      <div class="battle-lobby-summary">
        <strong>{{ room.full_pattern }}</strong>
        <span>{{ $t(`battle.status.${room.status}`) }}</span>
        <time v-if="room.expires_at && !permanentRoom">{{ remainingLabel }}</time>
      </div>
    </header>

    <div class="battle-lobby-grid">
      <section class="battle-seats-panel">
        <div class="battle-panel-title">
          <div><h3>{{ $t('battle.lobby.players') }}</h3><p>{{ players.length }}/{{ room.max_players }}</p></div>
        </div>

        <div class="battle-seat-grid">
          <article v-for="seat in seatRows" :key="seat.index" :class="['battle-seat', seat.member ? 'occupied' : 'empty']">
            <template v-if="seat.member">
              <div class="battle-seat-avatar">
                <img v-if="seat.member.avatar_url && !isBattleGuest(seat.member)" :src="seat.member.avatar_url" alt="" />
                <span v-else>{{ initials(seat.member.display_name) }}</span>
                <i :class="seat.member.online ? 'online' : 'offline'" aria-hidden="true"></i>
              </div>
              <div class="battle-seat-identity">
                <strong>{{ seat.member.display_name }} <small v-if="sameBattleActor(seat.member, selfMember)" class="battle-self-marker">{{ $t('battle.lobby.you') }}</small><small v-if="isBattleGuest(seat.member)" class="battle-guest-marker">{{ $t('battle.guest.marker') }}</small></strong>
                <span>{{ memberIsHost(seat.member) ? $t('battle.roles.host') : (seat.member.ready ? $t('battle.status.ready') : $t('battle.status.not_ready')) }}</span>
              </div>
              <button v-if="isHost && !sameBattleActor(seat.member, selfMember) && ['preparing', 'waiting'].includes(room.status)" type="button" class="battle-kick-btn" :title="$t('battle.actions.kick')" @click="$emit('kick', seat.member)">×</button>
            </template>
            <template v-else>
              <span class="battle-empty-seat-number">{{ String(seat.index + 1).padStart(2, '0') }}</span>
              <span>{{ $t('battle.lobby.openSeat') }}</span>
            </template>
          </article>
        </div>

        <p v-if="selfMember" class="battle-self-role" role="status">{{ $t('battle.lobby.yourRole') }} <strong>{{ $t(`battle.roles.${isHost ? 'host' : selfMember.role}`) }}</strong></p>
        <div class="battle-lobby-actions" :aria-busy="rolePending">
          <button v-if="selfMember?.role === 'player'" type="button" :class="['battle-ready-btn', selfMember.ready ? 'ready' : '']" :disabled="room.status !== 'waiting' || rolePending" @click="$emit('ready', !selfMember.ready)">
            {{ selfMember.ready ? $t('battle.actions.cancelReady') : $t('battle.actions.ready') }}
          </button>
          <button v-if="selfMember && !isHost && canChangeRole" type="button" class="battle-role-btn" :disabled="rolePending || (selfMember.role === 'spectator' ? seatsFull : !room.allow_spectators)" @click="$emit('role', selfMember.role === 'spectator' ? 'player' : 'spectator')">
            {{ $t(rolePending ? 'battle.actions.processing' : selfMember.role === 'spectator' ? (seatsFull ? 'battle.lobby.seatsFull' : 'battle.lobby.joinPlayer') : 'battle.lobby.becomeSpectator') }}
          </button>
          <button v-if="isHost" type="button" class="battle-start-btn" :disabled="!canStart" @click="$emit('start')">{{ $t('battle.actions.start') }}</button>
          <button type="button" class="battle-leave-btn" @click="$emit('leave')">{{ isHost && !permanentRoom ? $t('battle.actions.closeRoom') : $t('battle.actions.leave') }}</button>
        </div>
        <p class="battle-start-rule">{{ $t(room.allow_spectators ? 'battle.lobby.startRuleSpectators' : 'battle.lobby.startRuleNoSpectators') }}</p>
      </section>

      <aside class="battle-room-sidebar">
        <section class="battle-settings-summary">
          <div class="battle-panel-title">
            <div><h3>{{ $t('battle.lobby.settings') }}</h3><p>{{ settingsHint }}</p></div>
            <button v-if="canEditSettings && !editingSettings" type="button" class="battle-settings-edit" @click="beginSettingsEdit">{{ $t('battle.actions.edit') }}</button>
          </div>

          <div v-if="permanentRoom" :class="['battle-host-lease', { warning: hostLeaseWarning }]">
            <div>
              <span>{{ $t('battle.lobby.currentHost') }}</span>
              <strong>{{ room.host?.display_name || $t('battle.room.waitingHost') }}</strong>
            </div>
            <div v-if="room.host && room.host_idle_expires_at">
              <span>{{ $t('battle.lobby.hostIdleTime') }}</span>
              <strong class="tabular-nums">{{ hostIdleLabel }}</strong>
            </div>
            <p v-if="hostLeaseWarning" class="battle-host-warning">{{ $t('battle.lobby.hostIdleWarning') }}</p>
            <button
              v-if="isHost && hostLeaseWarning"
              type="button"
              :disabled="hostRenewPending"
              @click="$emit('renew-host')"
            >{{ $t(hostRenewPending ? 'battle.actions.processing' : 'battle.actions.continueHosting') }}</button>
          </div>

          <form v-if="editingSettings" class="battle-settings-form" @submit.prevent="saveSettings">
            <label class="battle-settings-field">
              <span>{{ $t('battle.form.stepTimeout') }}</span>
              <BattleNumberInput v-model="settingsDraft.step_timeout_seconds" :min="5" :max="600" :step="5" />
            </label>
            <template v-if="freeGoodnessRoom">
              <label class="battle-settings-field battle-settings-wide">
                <span>{{ $t('battle.form.initialBoard') }}</span>
                <input v-model.trim="settingsDraft.initial_board" maxlength="16" class="battle-settings-input font-mono" autocomplete="off" spellcheck="false" />
              </label>
              <label class="battle-settings-field">
                <span>{{ $t('battle.form.scoredSteps') }}</span>
                <BattleNumberInput v-model="settingsDraft.score_step_limit" :min="1" :max="targetStepCap" :step="1" />
              </label>
              <label class="battle-settings-field">
                <span>{{ $t('battle.form.rankingMinSteps') }}</span>
                <BattleNumberInput v-model="settingsDraft.ranking_min_steps" :min="1" :max="Math.max(1, Number(settingsDraft.score_step_limit) || 1)" :step="1" />
              </label>
            </template>
            <p v-if="settingsValidationError" class="battle-settings-error">{{ $t(`battle.settingsValidation.${settingsValidationError}`) }}</p>
            <div class="battle-settings-actions">
              <button type="button" :disabled="settingsPending" @click="cancelSettingsEdit">{{ $t('battle.actions.cancel') }}</button>
              <button type="submit" class="primary" :disabled="settingsPending">{{ $t(settingsPending ? 'battle.actions.processing' : 'battle.actions.save') }}</button>
            </div>
          </form>

          <slot v-else name="settings-summary" :room="room">
            <dl>
              <div><dt>{{ $t('battle.form.pattern') }}</dt><dd>{{ room.full_pattern }}</dd></div>
              <div><dt>{{ $t('battle.form.initialBoard') }}</dt><dd class="font-mono">{{ displayedInitialBoard }}</dd></div>
              <div><dt>{{ $t(stepsLabelKey) }}</dt><dd>{{ displayedStepLimit }}</dd></div>
              <div v-if="freeGoodnessRoom"><dt>{{ $t('battle.form.rankingMinSteps') }}</dt><dd>{{ displayedRankingMinSteps }}</dd></div>
              <div><dt>{{ $t('battle.form.stepTimeout') }}</dt><dd>{{ room.step_timeout_seconds }}s</dd></div>
              <div><dt>{{ $t('battle.form.publicRoom') }}</dt><dd>{{ room.visibility === 'public' ? $t('common.yes') : $t('common.no') }}</dd></div>
              <div><dt>{{ $t('battle.form.allowSpectators') }}</dt><dd>{{ room.allow_spectators ? $t('common.yes') : $t('common.no') }}</dd></div>
              <div><dt>{{ $t('battle.form.allowGuestChat') }}</dt><dd>{{ room.allow_guest_chat ? $t('common.yes') : $t('common.no') }}</dd></div>
              <div><dt>{{ $t('battle.form.chatRoles') }}</dt><dd>{{ chatRolesLabel }}</dd></div>
            </dl>
          </slot>
        </section>
        <section class="battle-spectator-list">
          <div class="battle-panel-title"><div><h3>{{ $t('battle.lobby.spectators') }}</h3><p>{{ spectators.length }}</p></div></div>
          <div v-if="!spectators.length" class="battle-sidebar-empty">{{ $t('battle.lobby.noSpectators') }}</div>
          <div v-else class="battle-spectator-chips">
            <span v-for="member in spectators" :key="battleActorRenderKey(member)">{{ member.display_name }}<small v-if="sameBattleActor(member, selfMember)" class="battle-self-marker">{{ $t('battle.lobby.you') }}</small><small v-if="isBattleGuest(member)" class="battle-guest-marker">{{ $t('battle.guest.marker') }}</small></span>
          </div>
        </section>
      </aside>
    </div>
  </div>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import {
  battleActorRenderKey,
  isBattleGuest,
  sameBattleActor,
} from '../core/battleActor.js';
import {
  battleRoomSettingsDraft,
  buildBattleRoomSettingsPayload,
  isPermanentBattleRoom,
  validateBattleRoomSettings,
} from '../core/battleRoomSettings.js';
import BattleNumberInput from './BattleNumberInput.vue';

const props = defineProps({
  room: { type: Object, required: true },
  members: { type: Array, default: () => [] },
  currentUserId: { type: Number, default: 0 },
  currentActorKey: { type: String, default: '' },
  now: { type: Number, default: () => Date.now() },
  settingsPending: { type: Boolean, default: false },
  rolePending: { type: Boolean, default: false },
  hostRenewPending: { type: Boolean, default: false },
});

const emit = defineEmits(['ready', 'start', 'kick', 'leave', 'role', 'save-settings', 'renew-host']);
const { t } = useI18n();
const copied = ref(false);
const editingSettings = ref(false);
const settingsValidationError = ref('');
const settingsRevision = ref(0);
const settingsSubmitted = ref(false);
const settingsDraft = reactive(battleRoomSettingsDraft(props.room));
const players = computed(() => props.members.filter((member) => member.role === 'player' && member.status === 'active'));
const spectators = computed(() => props.members.filter((member) => member.role === 'spectator' && member.status === 'active'));
const selfMember = computed(() => props.members.find((member) => (
  (props.currentActorKey && battleActorRenderKey(member) === props.currentActorKey)
  || (!props.currentActorKey && Number(member.user_id) === Number(props.currentUserId))
)) || null);
const isHost = computed(() => Boolean(props.room.viewer?.is_host));
const canChangeRole = computed(() => ['preparing', 'waiting'].includes(props.room.status));
const seatsFull = computed(() => players.value.length >= Number(props.room.max_players));
const permanentRoom = computed(() => isPermanentBattleRoom(props.room));
const freeGoodnessRoom = computed(() => String(props.room.mode_key || '') === 'free_goodness');
const canEditSettings = computed(() => (
  isHost.value && ['preparing', 'waiting'].includes(String(props.room.status || ''))
));
const canStart = computed(() => (
  isHost.value
  && Boolean(selfMember.value?.ready)
  && props.room.status === 'waiting'
));
const seatRows = computed(() => Array.from({ length: Number(props.room.max_players || 2) }, (_unused, index) => ({ index, member: players.value[index] || null })));
const displayedInitialBoard = computed(() => (
  battleRoomSettingsDraft(props.room).initial_board
  || t('battle.status.preparingInitialBoard')
));
const stepsLabelKey = computed(() => (
  freeGoodnessRoom.value
    ? 'battle.form.scoredSteps'
    : 'battle.form.maxSteps'
));
const displayedStepLimit = computed(() => (
  freeGoodnessRoom.value
    ? battleRoomSettingsDraft(props.room).score_step_limit
    : (props.room.max_steps || t('battle.form.unlimited'))
));
const displayedRankingMinSteps = computed(() => battleRoomSettingsDraft(props.room).ranking_min_steps);
const targetStepCap = computed(() => Math.max(1, Math.floor(Number(props.room.target || 0) / 2)));
const settingsHint = computed(() => (
  canEditSettings.value ? t('battle.lobby.settingsEditable') : t('battle.lobby.settingsReadOnly')
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
const hostIdleRemaining = computed(() => Math.max(
  0,
  Date.parse(props.room.host_idle_expires_at || '') - Number(props.now),
));
const hostLeaseWarning = computed(() => (
  Boolean(props.room.host_idle_expires_at) && hostIdleRemaining.value <= 30_000
));
const hostIdleLabel = computed(() => {
  const totalSeconds = Math.ceil(hostIdleRemaining.value / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
});
const initials = (name) => String(name || '?').trim().slice(0, 2).toUpperCase();
const memberIsHost = (member) => (
  Boolean(props.room.host_actor_key)
  && battleActorRenderKey(member) === String(props.room.host_actor_key)
);
const syncSettingsDraft = () => Object.assign(settingsDraft, battleRoomSettingsDraft(props.room));
const beginSettingsEdit = () => {
  if (!canEditSettings.value) return;
  syncSettingsDraft();
  settingsRevision.value = Number(props.room.settings_revision || 0);
  settingsValidationError.value = '';
  settingsSubmitted.value = false;
  editingSettings.value = true;
};
const cancelSettingsEdit = () => {
  editingSettings.value = false;
  settingsValidationError.value = '';
  settingsSubmitted.value = false;
  syncSettingsDraft();
};
const saveSettings = () => {
  if (!canEditSettings.value || props.settingsPending) return;
  const validation = validateBattleRoomSettings(props.room, settingsDraft);
  if (!validation.ok) {
    settingsValidationError.value = validation.code;
    return;
  }
  settingsValidationError.value = '';
  settingsSubmitted.value = true;
  emit('save-settings', buildBattleRoomSettingsPayload(
    props.room,
    settingsDraft,
    settingsRevision.value,
  ));
};
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

watch(() => props.room.settings_revision, (revision) => {
  if (settingsSubmitted.value && Number(revision) !== settingsRevision.value) {
    cancelSettingsEdit();
    return;
  }
  if (!editingSettings.value) syncSettingsDraft();
});
watch(() => props.room.room_id, () => cancelSettingsEdit());
watch(canEditSettings, (editable) => {
  if (!editable && editingSettings.value) cancelSettingsEdit();
});
</script>

<style scoped>
.battle-lobby { display: flex; flex-direction: column; gap: 16px; }
.battle-lobby-banner { min-height: 92px; display: flex; align-items: center; justify-content: space-between; gap: 20px; padding: 17px 20px; border: 1px solid var(--border-main); border-radius: 8px; background: var(--bg-card); box-shadow: 0 14px 32px rgba(0,0,0,.07); }
.battle-room-code-line { display: flex; align-items: center; gap: 12px; margin-top: 4px; }
.battle-room-code-line h2 { margin: 0; color: var(--text-main); font: 900 28px/1 var(--font-mono, monospace); letter-spacing: .08em; }
.battle-lobby-permanent { padding: 4px 6px; border: 1px solid color-mix(in srgb, var(--accent) 55%, var(--border-main)); border-radius: 4px; color: var(--accent); font-size: 9px; font-weight: 900; line-height: 1; }
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
.battle-settings-edit { min-height: 30px; padding: 5px 10px; border: 1px solid var(--border-main); border-radius: 6px; background: var(--bg-main); color: var(--text-main); font-size: var(--font-ui-xs); font-weight: 900; }
.battle-settings-edit:hover { border-color: var(--accent); color: var(--accent); }
.battle-host-lease { display: grid; grid-template-columns: 1fr auto; gap: 8px 12px; margin-bottom: 12px; padding: 10px; border: 1px solid var(--border-main); border-radius: 7px; background: color-mix(in srgb, var(--bg-main) 72%, transparent); }
.battle-host-lease.warning { border-color: color-mix(in srgb, #dc8c32 62%, var(--border-main)); background: color-mix(in srgb, #dc8c32 8%, var(--bg-main)); }
.battle-host-lease > div { min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.battle-host-lease span { color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 800; }
.battle-host-lease strong { overflow: hidden; color: var(--text-main); font-size: var(--font-ui-sm); text-overflow: ellipsis; white-space: nowrap; }
.battle-host-warning { grid-column: 1 / -1; margin: 0; color: #b66e1f; font-size: var(--font-ui-xs); font-weight: 900; }
.battle-host-lease button { grid-column: 1 / -1; min-height: 32px; border: 1px solid color-mix(in srgb, #dc8c32 62%, var(--border-main)); border-radius: 6px; background: transparent; color: #b66e1f; font-size: var(--font-ui-xs); font-weight: 900; }
.battle-host-lease button:disabled { opacity: .45; }
.battle-settings-form { display: grid; grid-template-columns: 1fr 1fr; gap: 11px 9px; }
.battle-settings-field { min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.battle-settings-field > span { color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 900; }
.battle-settings-wide { grid-column: 1 / -1; }
.battle-settings-input { width: 100%; min-width: 0; min-height: 38px; padding: 8px 10px; border: 1px solid var(--border-main); border-radius: 7px; background: var(--bg-main); color: var(--text-main); font-size: var(--font-ui-sm); font-weight: 900; outline: none; }
.battle-settings-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 15%, transparent); }
.battle-settings-error { grid-column: 1 / -1; margin: 0; color: #c84848; font-size: var(--font-ui-xs); font-weight: 800; }
.battle-settings-actions { grid-column: 1 / -1; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding-top: 2px; }
.battle-settings-actions button { min-height: 36px; border: 1px solid var(--border-main); border-radius: 7px; background: var(--bg-main); color: var(--text-main); font-size: var(--font-ui-xs); font-weight: 900; }
.battle-settings-actions button.primary { border-color: var(--btn-bg); background: var(--btn-bg); color: white; }
.battle-settings-actions button:disabled { opacity: .42; }
.battle-self-role { margin: 16px 0 8px; color: var(--text-secondary); font-size: var(--font-ui-xs); }
.battle-self-role strong { margin-left: 5px; color: var(--text-main); }
.battle-self-marker { margin-left: 4px; color: var(--accent); font-size: var(--font-ui-xs); }
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
.battle-guest-marker { display: inline-block; margin-left: 4px; color: var(--accent); font-size: 8px; font-weight: 900; vertical-align: 1px; }
.battle-seat-identity span { color: var(--text-secondary); font-size: var(--font-ui-xs); }
.battle-kick-btn { width: 28px; height: 28px; padding: 0; color: #dc4c4c; }
.battle-lobby-actions { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 8px; }
.battle-lobby-actions > button { flex: 1 1 130px; min-width: 0; padding: 8px 14px; overflow-wrap: anywhere; }
.battle-ready-btn, .battle-start-btn, .battle-leave-btn, .battle-role-btn { min-height: 42px; border: 1px solid var(--border-main); border-radius: 7px; font-size: var(--font-ui-sm); font-weight: 900; }
.battle-role-btn { background: var(--bg-main); color: var(--text-main); }
.battle-lobby-actions > button:disabled { opacity: .4; cursor: not-allowed; }
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
.battle-spectator-chips { display: flex; flex-wrap: wrap; gap: 7px; max-height: 180px; overflow-y: auto; align-content: flex-start; }
.battle-spectator-chips span { max-width: 100%; overflow: hidden; text-overflow: ellipsis; border: 1px solid var(--border-main); border-radius: 999px; padding: 5px 9px; color: var(--text-main); font-size: var(--font-ui-xs); }
</style>
