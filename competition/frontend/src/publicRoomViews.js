import DuelSetup from './DuelSetup.vue';
import TimeAttackSetup from './TimeAttackSetup.vue';
// Mode-specific configuration views only; the common shell owns submission.
export const PUBLIC_ROOM_SETUPS = Object.freeze({duel:DuelSetup,time_attack:TimeAttackSetup});
