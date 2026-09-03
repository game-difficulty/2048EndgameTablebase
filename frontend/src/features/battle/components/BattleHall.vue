<template>
  <div class="battle-hall-grid">
    <section class="battle-hall-list" aria-labelledby="battle-room-list-title">
      <header class="battle-section-header">
        <div>
          <div class="ui-caption font-black uppercase text-text-secondary">{{ $t('battle.hall.kicker') }}</div>
          <h2 id="battle-room-list-title">{{ $t('battle.hall.modeRooms', { mode: selectedModeLabel }) }}</h2>
        </div>
        <button type="button" class="battle-icon-btn" :title="$t('common.refresh')" @click="$emit('refresh')">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M19 7v5h-5M5 17v-5h5M7.2 8.3A6.5 6.5 0 0 1 18.4 10M16.8 15.7A6.5 6.5 0 0 1 5.6 14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>
      </header>

      <div v-if="loading" class="battle-empty-state">{{ $t('common.loading') }}</div>
      <div v-else-if="!visibleRooms.length" class="battle-empty-state">
        <strong>{{ $t('battle.hall.emptyTitle') }}</strong>
        <span>{{ $t('battle.hall.emptyBody') }}</span>
      </div>
      <div v-else class="battle-room-table">
        <div class="battle-room-table-head">
          <span>{{ $t('battle.room.host') }}</span>
          <span>{{ $t('battle.room.tablebase') }}</span>
          <span>{{ $t('battle.room.players') }}</span>
          <span>{{ $t('battle.room.moveTime') }}</span>
          <span>{{ $t('battle.room.status') }}</span>
        </div>
        <article
          v-for="room in visibleRooms"
          :key="room.room_code"
          class="battle-room-row"
          role="button"
          tabindex="0"
          :aria-label="`${room.status === 'running' ? $t('battle.actions.spectate') : $t('battle.actions.join')} ${room.full_pattern}`"
          @click="joinRoom(room)"
          @keydown.enter="joinRoom(room)"
          @keydown.space.prevent="joinRoom(room)"
        >
          <div class="battle-room-host">
            <img v-if="room.host?.avatar_url" :src="room.host.avatar_url" alt="" />
            <span v-else class="battle-avatar-fallback">{{ initials(room.host?.display_name) }}</span>
            <strong>{{ room.host?.display_name || $t('battle.room.unknownHost') }}</strong>
          </div>
          <strong class="battle-room-pattern">{{ room.full_pattern }}</strong>
          <span class="tabular-nums">{{ room.player_count }}/{{ room.max_players }}</span>
          <span class="tabular-nums">{{ room.step_timeout_seconds }}s</span>
          <span :class="['battle-status-dot-label', `status-${room.status}`]">
            <i aria-hidden="true"></i>{{ $t(`battle.status.${room.status}`) }}
          </span>
        </article>
      </div>
    </section>

    <aside class="battle-hall-tools">
      <section class="battle-mode-switcher" aria-labelledby="battle-mode-switcher-title">
        <div>
          <span class="ui-caption font-black uppercase text-text-secondary">{{ $t('battle.hall.modeKicker') }}</span>
          <strong id="battle-mode-switcher-title">{{ $t('battle.form.mode') }}</strong>
        </div>
        <UiSelect
          :model-value="modeKey"
          :options="modeOptions"
          trigger-class="battle-mode-select-trigger"
          align="right"
          @change="changeMode"
        />
      </section>

      <section class="battle-tool-section">
        <div class="battle-tool-heading">
          <span class="battle-tool-index">01</span>
          <div>
            <h2>{{ $t('battle.hall.joinByCode') }}</h2>
            <p>{{ $t('battle.hall.joinByCodeHint') }}</p>
          </div>
        </div>
        <div class="battle-code-row">
          <input
            v-model="joinCode"
            class="battle-code-input"
            maxlength="6"
            autocomplete="off"
            spellcheck="false"
            :placeholder="$t('battle.hall.roomCode')"
            @input="normalizeCode"
            @keydown.enter="submitJoin"
          />
          <button type="button" class="battle-command-btn prominent" :disabled="joinCode.length !== 6" @click="submitJoin">
            {{ $t('battle.actions.join') }}
          </button>
        </div>
      </section>

      <section class="battle-tool-section battle-create-section">
        <div class="battle-tool-heading">
          <span class="battle-tool-index">02</span>
          <div>
            <h2>{{ $t('battle.hall.createRoom') }}</h2>
            <p>{{ $t('battle.hall.createHint') }}</p>
          </div>
        </div>

        <div class="battle-form-grid">
          <slot
            name="mode-settings"
            :form="form"
            :pattern-options="patternOptions"
            :target-options="targetOptions"
            :sync-target="syncTarget"
          >
            <label class="battle-field">
              <span>{{ $t('battle.form.pattern') }}</span>
              <UiSelect v-model="form.pattern" :options="patternOptions" trigger-class="battle-select-trigger" @change="syncTarget" />
            </label>
            <label class="battle-field">
              <span>{{ $t('battle.form.target') }}</span>
              <UiSelect v-model="form.target" :options="targetOptions" trigger-class="battle-select-trigger" />
            </label>
            <label class="battle-field battle-field-wide">
              <span>{{ $t('battle.form.initialBoard') }}</span>
              <input v-model.trim="form.initial_board" maxlength="16" class="battle-text-input font-mono" :placeholder="$t('battle.form.randomBoard')" />
            </label>
          </slot>
          <label class="battle-field">
            <span>{{ $t('battle.form.maxPlayers') }}</span>
            <BattleNumberInput v-model="form.max_players" :min="2" :max="8" :step="1" />
          </label>
          <label class="battle-field">
            <span>{{ $t('battle.form.stepTimeout') }}</span>
            <BattleNumberInput v-model="form.step_timeout_seconds" :min="5" :max="600" :step="5" />
          </label>
          <label v-if="showMaxSteps" class="battle-field battle-field-wide">
            <span>{{ $t('battle.form.maxSteps') }}</span>
            <BattleNumberInput v-model="form.max_steps" :min="1" :max="9999" :step="1" :placeholder="$t('battle.form.unlimited')" allow-empty />
          </label>
          <label v-if="showRankingMinSteps" class="battle-field battle-field-wide">
            <span>{{ $t('battle.form.rankingMinSteps') }}</span>
            <BattleNumberInput
              v-model="form.ranking_min_steps"
              :min="1"
              :max="rankingStepCap"
              :step="1"
              :placeholder="$t('battle.form.rankingMinStepsDefault', { count: rankingStepCap })"
              allow-empty
            />
          </label>
        </div>

        <div class="battle-toggle-list">
          <fieldset class="battle-chat-role-fieldset">
            <legend>{{ $t('battle.form.chatRoles') }}</legend>
            <small>{{ $t('battle.form.chatRolesHint') }}</small>
            <div class="battle-chat-role-options">
              <label v-for="role in chatRoleOptions" :key="role" :class="{ selected: form.chat_roles.includes(role) }">
                <input v-model="form.chat_roles" type="checkbox" :value="role" />
                <span>{{ $t(`battle.roles.${role}`) }}</span>
              </label>
            </div>
          </fieldset>
          <label class="battle-toggle-row">
            <span><strong>{{ $t('battle.form.publicRoom') }}</strong><small>{{ $t('battle.form.publicRoomHint') }}</small></span>
            <input v-model="form.is_public" type="checkbox" />
          </label>
          <label class="battle-toggle-row">
            <span><strong>{{ $t('battle.form.allowSpectators') }}</strong><small>{{ $t('battle.form.allowSpectatorsHint') }}</small></span>
            <input v-model="form.allow_spectators" type="checkbox" />
          </label>
          <label v-if="canCreateRoom" class="battle-toggle-row">
            <span><strong>{{ $t('battle.form.allowGuestChat') }}</strong><small>{{ $t('battle.form.allowGuestChatHint') }}</small></span>
            <input v-model="form.allow_guest_chat" type="checkbox" />
          </label>
        </div>

        <div class="battle-cost-row">
          <div><span>{{ $t(costLabelKey) }}</span><strong>{{ formattedCost }} Token</strong></div>
          <div><span>{{ $t('battle.form.balance') }}</span><strong>{{ formattedBalance }}</strong></div>
        </div>
        <p class="battle-refund-policy">{{ $t(refundPolicyKey) }}</p>
        <button type="button" class="battle-create-btn" :disabled="creating || (canCreateRoom && !canCreate)" @click="submitCreate">
          {{ creating ? $t('battle.status.preparing') : $t(createLabelKey) }}
        </button>
        <p v-if="!canCreateRoom" class="battle-guest-create-note">{{ $t('battle.guest.createRequiresLogin') }}</p>
      </section>
    </aside>
  </div>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue';

import UiSelect from '../../../components/UiSelect.vue';
import { getCatalogTargetsForPattern } from '../../../services/tablebases/catalogClient.js';
import BattleNumberInput from './BattleNumberInput.vue';

const props = defineProps({
  rooms: { type: Array, default: () => [] },
  tables: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
  creating: { type: Boolean, default: false },
  tokenBalance: { type: Number, default: 0 },
  multiplierForPattern: { type: Function, required: true },
  routeBaseCost: { type: Number, default: 100 },
  calculateCost: { type: Function, default: null },
  showMaxSteps: { type: Boolean, default: true },
  showRankingMinSteps: { type: Boolean, default: false },
  costLabelKey: { type: String, default: 'battle.form.routeCost' },
  refundPolicyKey: { type: String, default: 'battle.form.refundPolicy' },
  createLabelKey: { type: String, default: 'battle.actions.create' },
  modeKey: { type: String, default: 'goodness' },
  modeOptions: { type: Array, default: () => [] },
  buildCreatePayload: { type: Function, default: null },
  canCreateRoom: { type: Boolean, default: true },
});

const emit = defineEmits(['refresh', 'join', 'create', 'mode-change', 'login-required']);
const joinCode = ref('');
const form = reactive({
  pattern: '',
  target: '',
  initial_board: '',
  max_players: 2,
  max_steps: '',
  ranking_min_steps: '',
  step_timeout_seconds: 90,
  is_public: true,
  allow_spectators: true,
  allow_guest_chat: false,
  chat_roles: ['host', 'player', 'spectator'],
});

const chatRoleOptions = ['host', 'player', 'spectator'];

const patternOptions = computed(() => [...new Set(props.tables.map((table) => table.pattern).filter(Boolean))]);
const targetOptions = computed(() => getCatalogTargetsForPattern(props.tables, form.pattern));
const selectedFullPattern = computed(() => `${form.pattern}_${form.target}`);
const normalizedModeKey = (modeKey) => String(modeKey || 'goodness').trim().toLowerCase();
const visibleRooms = computed(() => props.rooms.filter(
  (room) => normalizedModeKey(room.mode_key) === normalizedModeKey(props.modeKey),
));
const selectedModeOption = computed(() => props.modeOptions.find(
  (option) => option.value === props.modeKey,
));
const selectedModeLabel = computed(() => (
  selectedModeOption.value?.label
  || selectedModeOption.value?.shortLabel
  || props.modeKey
));
const validMode = computed(() => Boolean(selectedModeOption.value));
const routeCost = computed(() => (
  props.calculateCost
    ? props.calculateCost({ ...form, full_pattern: selectedFullPattern.value })
    : props.routeBaseCost * props.multiplierForPattern(selectedFullPattern.value)
));
const formattedCost = computed(() => routeCost.value.toLocaleString());
const formattedBalance = computed(() => Number(props.tokenBalance || 0).toLocaleString(undefined, { maximumFractionDigits: 1 }));
const validInitialBoard = computed(() => !form.initial_board || /^[0-9a-fA-F]{16}$/.test(form.initial_board));
const validMaxPlayers = computed(() => Number.isInteger(Number(form.max_players)) && Number(form.max_players) >= 2 && Number(form.max_players) <= 8);
const validStepTimeout = computed(() => Number.isInteger(Number(form.step_timeout_seconds)) && Number(form.step_timeout_seconds) >= 5 && Number(form.step_timeout_seconds) <= 600 && Number(form.step_timeout_seconds) % 5 === 0);
const validMaxSteps = computed(() => form.max_steps === '' || (Number.isInteger(Number(form.max_steps)) && Number(form.max_steps) >= 1 && Number(form.max_steps) <= 9999));
const rankingStepCap = computed(() => Math.max(1, Math.floor((Number(form.target) || 0) / 2)));
const validRankingMinSteps = computed(() => (
  !props.showRankingMinSteps
  || form.ranking_min_steps === ''
  || (
    Number.isInteger(Number(form.ranking_min_steps))
    && Number(form.ranking_min_steps) >= 1
    && Number(form.ranking_min_steps) <= rankingStepCap.value
  )
));
const canCreate = computed(() => Boolean(validMode.value && form.pattern && form.target && validInitialBoard.value && validMaxPlayers.value && validStepTimeout.value && validMaxSteps.value && validRankingMinSteps.value));

const initials = (name) => String(name || '?').trim().slice(0, 2).toUpperCase();
const changeMode = (modeKey) => emit('mode-change', modeKey);
const joinRoom = (room) => {
  if (!room || props.loading) return;
  emit('join', room.room_code, room.player_count >= room.max_players ? 'spectator' : 'auto');
};
const normalizeCode = () => {
  joinCode.value = joinCode.value.toUpperCase().replace(/[^2-9A-HJ-NP-Z]/g, '').slice(0, 6);
};
const syncTarget = () => {
  if (!targetOptions.value.includes(form.target)) form.target = targetOptions.value[0] || '';
};
const submitJoin = () => {
  normalizeCode();
  if (joinCode.value.length === 6) emit('join', joinCode.value, 'auto');
};
const submitCreate = () => {
  if (!props.canCreateRoom) {
    emit('login-required');
    return;
  }
  if (!canCreate.value) return;
  const defaultPayload = {
    ...form,
    mode_key: props.modeKey,
    full_pattern: selectedFullPattern.value,
    initial_board: form.initial_board || null,
    max_steps: form.max_steps === '' ? null : Number(form.max_steps),
    ranking_min_steps: form.ranking_min_steps === '' ? null : Number(form.ranking_min_steps),
  };
  emit('create', props.buildCreatePayload
    ? props.buildCreatePayload({ ...defaultPayload })
    : defaultPayload);
};

watch(patternOptions, (options) => {
  if (!options.includes(form.pattern)) form.pattern = options[0] || '';
  syncTarget();
}, { immediate: true });
watch(rankingStepCap, (cap) => {
  if (form.ranking_min_steps !== '' && Number(form.ranking_min_steps) > cap) {
    form.ranking_min_steps = cap;
  }
});
</script>

<style scoped>
.battle-hall-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.65fr) minmax(350px, 0.85fr);
  gap: 18px;
  min-height: 640px;
}

.battle-hall-list,
.battle-hall-tools {
  min-width: 0;
}

.battle-hall-list,
.battle-mode-switcher,
.battle-tool-section {
  border: 1px solid var(--border-main);
  background: color-mix(in srgb, var(--bg-card) 95%, transparent);
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.07);
}

.battle-hall-list {
  border-radius: 8px;
  overflow: hidden;
}

.battle-section-header {
  min-height: 82px;
  padding: 18px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  border-bottom: 1px solid var(--border-main);
}

.battle-section-header h2,
.battle-tool-heading h2 {
  margin: 3px 0 0;
  color: var(--text-main);
  font-size: var(--font-ui-lg);
  font-weight: 900;
  letter-spacing: 0;
}

.battle-icon-btn {
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  border: 1px solid var(--border-main);
  border-radius: 8px;
  color: var(--text-main);
  background: var(--bg-main);
}

.battle-icon-btn svg { width: 18px; height: 18px; }
.battle-icon-btn:hover { color: var(--accent); border-color: var(--accent); }

.battle-empty-state {
  min-height: 440px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--text-secondary);
  font-size: var(--font-ui-sm);
}

.battle-empty-state strong { color: var(--text-main); font-size: var(--font-ui-base); }
.battle-room-table { padding: 0 12px 12px; }

.battle-room-table-head,
.battle-room-row {
  display: grid;
  grid-template-columns: minmax(130px, 1.1fr) minmax(112px, 1fr) 68px 66px 108px;
  align-items: center;
  gap: 12px;
}

.battle-room-table-head {
  padding: 13px 10px 9px;
  color: var(--text-secondary);
  font-size: var(--font-ui-xs);
  font-weight: 900;
  text-transform: uppercase;
}

.battle-room-row {
  min-height: 62px;
  padding: 10px;
  border-top: 1px solid color-mix(in srgb, var(--border-main) 76%, transparent);
  color: var(--text-main);
  font-size: var(--font-ui-sm);
  cursor: pointer;
  outline: none;
}

.battle-room-row:hover,
.battle-room-row:focus-visible {
  background: color-mix(in srgb, var(--bg-main) 65%, transparent);
}
.battle-room-row:focus-visible {
  box-shadow: inset 0 0 0 2px color-mix(in srgb, var(--accent) 72%, transparent);
}
.battle-room-host { display: flex; align-items: center; gap: 9px; min-width: 0; }
.battle-room-host img, .battle-avatar-fallback { width: 34px; height: 34px; border-radius: 50%; flex: 0 0 auto; }
.battle-room-host img { object-fit: cover; }
.battle-avatar-fallback { display: grid; place-items: center; border: 1px solid var(--border-main); color: var(--accent); font-size: 11px; font-weight: 900; }
.battle-room-host strong, .battle-room-pattern { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.battle-status-dot-label { display: inline-flex; align-items: center; gap: 7px; color: var(--text-secondary); font-weight: 800; }
.battle-status-dot-label i { width: 7px; height: 7px; border-radius: 50%; background: #8e98a7; }
.battle-status-dot-label.status-waiting i { background: #37a667; }
.battle-status-dot-label.status-running i { background: #e1a92f; }
.battle-status-dot-label.status-preparing i { background: #3b9ed8; }

.battle-command-btn,
.battle-create-btn {
  border: 1px solid var(--border-main);
  border-radius: 7px;
  background: var(--bg-main);
  color: var(--text-main);
  font-size: var(--font-ui-sm);
  font-weight: 900;
  min-height: 36px;
  padding: 7px 13px;
}

.battle-command-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
.battle-command-btn.prominent, .battle-create-btn { background: var(--btn-bg); border-color: var(--btn-bg); color: white; }
.battle-command-btn:disabled, .battle-create-btn:disabled { opacity: 0.42; cursor: not-allowed; }

.battle-hall-tools { display: flex; flex-direction: column; gap: 14px; }
.battle-mode-switcher { min-height: 70px; display: grid; grid-template-columns: minmax(0, 1fr) minmax(150px, 0.9fr); align-items: center; gap: 14px; padding: 13px 16px; border-radius: 8px; }
.battle-mode-switcher > div { display: flex; min-width: 0; flex-direction: column; gap: 4px; }
.battle-mode-switcher strong { color: var(--text-main); font-size: var(--font-ui-sm); font-weight: 900; }
:deep(.battle-mode-select-trigger) { min-height: 40px; padding: 8px 10px; border: 1px solid var(--border-main); border-radius: 7px; background: var(--bg-main); color: var(--text-main); font-size: var(--font-ui-sm); font-weight: 900; }
.battle-tool-section { border-radius: 8px; padding: 18px; }
.battle-create-section { flex: 1; }
.battle-tool-heading { display: flex; gap: 12px; align-items: flex-start; margin-bottom: 16px; }
.battle-tool-heading p { margin: 4px 0 0; color: var(--text-secondary); font-size: var(--font-ui-xs); line-height: 1.45; }
.battle-tool-index { color: var(--accent); font: 900 12px/1 var(--font-mono, monospace); padding-top: 6px; }
.battle-code-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; }
.battle-code-input, .battle-text-input {
  min-width: 0;
  min-height: 38px;
  padding: 8px 11px;
  border: 1px solid var(--border-main);
  border-radius: 7px;
  background: var(--bg-main);
  color: var(--text-main);
  outline: none;
}
.battle-code-input { text-align: center; font: 900 16px/1 var(--font-mono, monospace); letter-spacing: 0.14em; }
.battle-code-input:focus, .battle-text-input:focus { border-color: var(--accent); }
.battle-form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 10px; }
.battle-field { display: flex; min-width: 0; flex-direction: column; gap: 6px; }
.battle-field > span { color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 900; }
.battle-field-wide { grid-column: 1 / -1; }
:deep(.battle-select-trigger) { min-height: 38px; padding: 8px 10px; border: 1px solid var(--border-main); border-radius: 7px; background: var(--bg-main); color: var(--text-main); font-size: var(--font-ui-sm); font-weight: 900; }
.battle-toggle-list { margin-top: 14px; border-top: 1px solid var(--border-main); }
.battle-chat-role-fieldset { margin: 0; padding: 11px 2px 13px; border: 0; border-bottom: 1px solid color-mix(in srgb, var(--border-main) 72%, transparent); }
.battle-chat-role-fieldset legend { padding: 0; color: var(--text-main); font-size: var(--font-ui-sm); font-weight: 900; }
.battle-chat-role-fieldset > small { display: block; margin-top: 3px; color: var(--text-secondary); font-size: var(--font-ui-xs); }
.battle-chat-role-options { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; margin-top: 9px; }
.battle-chat-role-options label { min-width: 0; min-height: 34px; display: flex; align-items: center; justify-content: center; gap: 7px; padding: 6px 8px; border: 1px solid var(--border-main); border-radius: 7px; background: var(--bg-main); color: var(--text-main); font-size: var(--font-ui-xs); font-weight: 900; }
.battle-chat-role-options label.selected { border-color: var(--accent); color: var(--accent); }
.battle-chat-role-options input { width: 15px; height: 15px; margin: 0; accent-color: var(--accent); }
.battle-toggle-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 11px 2px; border-bottom: 1px solid color-mix(in srgb, var(--border-main) 72%, transparent); }
.battle-toggle-row span { display: flex; flex-direction: column; gap: 3px; }
.battle-toggle-row strong { color: var(--text-main); font-size: var(--font-ui-sm); }
.battle-toggle-row small { color: var(--text-secondary); font-size: var(--font-ui-xs); }
.battle-toggle-row input { width: 18px; height: 18px; accent-color: var(--accent); }
.battle-cost-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
.battle-cost-row div { display: flex; flex-direction: column; gap: 3px; padding: 10px; border: 1px solid var(--border-main); border-radius: 7px; background: color-mix(in srgb, var(--bg-main) 70%, transparent); }
.battle-cost-row span { color: var(--text-secondary); font-size: var(--font-ui-xs); }
.battle-cost-row strong { color: var(--text-main); font-size: var(--font-ui-sm); }
.battle-refund-policy { margin: 8px 2px 0; color: var(--text-secondary); font-size: var(--font-ui-xs); line-height: 1.45; }
.battle-create-btn { width: 100%; min-height: 42px; margin-top: 12px; }
.battle-guest-create-note { margin: 8px 0 0; color: var(--text-secondary); font-size: var(--font-ui-xs); font-weight: 700; text-align: center; }

@media (max-width: 1100px) {
  .battle-hall-grid { grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.8fr); }
  .battle-room-table-head, .battle-room-row { grid-template-columns: minmax(120px, 1fr) minmax(110px, 0.9fr) 68px 92px 74px; gap: 8px; }
}
</style>
