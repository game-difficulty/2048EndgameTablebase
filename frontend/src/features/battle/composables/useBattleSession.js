import { computed, ref } from 'vue';

import { useBattleRoomSession } from '../core/useBattleRoomSession.js';
import {
  getBattleModeDefinition,
  listBattleModeDefinitions,
} from '../core/modeRegistry.js';


export function useBattleSession(
  activeRef,
  authUserRef,
  hotkeysEnabledRef = activeRef,
  actorRef = authUserRef,
  ensureGuestSession = null,
) {
  const roomSession = useBattleRoomSession(
    activeRef,
    authUserRef,
    actorRef,
    ensureGuestSession,
  );
  const defaultDefinition = getBattleModeDefinition('goodness');
  const modeDefinitions = listBattleModeDefinitions();
  const selectedModeKey = ref(defaultDefinition.key);
  const sessions = new Map(
    modeDefinitions.map((definition) => [
      definition.key,
      definition.createSession(roomSession, activeRef, authUserRef, hotkeysEnabledRef),
    ]),
  );
  const modeDefinition = computed(() => getBattleModeDefinition(
    roomSession.room.value?.mode_key || selectedModeKey.value,
  ));
  const modeSession = computed(() => (
    sessions.get(modeDefinition.value.key) || sessions.get(defaultDefinition.key)
  ));
  const selectMode = (modeKey) => {
    const requested = String(modeKey || '').trim().toLowerCase();
    if (modeDefinitions.some((definition) => definition.key === requested)) {
      selectedModeKey.value = requested;
      void roomSession.bootstrapMode(requested);
    }
  };

  return {
    ...roomSession,
    modeDefinitions,
    selectedModeKey,
    selectMode,
    modeDefinition,
    modeSession,
  };
}
