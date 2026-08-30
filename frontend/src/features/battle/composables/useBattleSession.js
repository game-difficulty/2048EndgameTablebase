import { computed } from 'vue';

import { useBattleRoomSession } from '../core/useBattleRoomSession.js';
import {
  getBattleModeDefinition,
  listBattleModeDefinitions,
} from '../core/modeRegistry.js';


export function useBattleSession(activeRef, authUserRef) {
  const roomSession = useBattleRoomSession(activeRef, authUserRef);
  const defaultDefinition = getBattleModeDefinition('goodness');
  const sessions = new Map(
    listBattleModeDefinitions().map((definition) => [
      definition.key,
      definition.createSession(roomSession, activeRef, authUserRef),
    ]),
  );
  const modeDefinition = computed(() => getBattleModeDefinition(
    roomSession.room.value?.mode_key || defaultDefinition.key,
  ));
  const modeSession = computed(() => (
    sessions.get(modeDefinition.value.key) || sessions.get(defaultDefinition.key)
  ));

  return {
    ...roomSession,
    modeDefinition,
    modeSession,
  };
}
