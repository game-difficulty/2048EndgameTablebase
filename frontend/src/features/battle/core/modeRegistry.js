import BattleHall from '../components/BattleHall.vue';
import BattleMatch from '../components/BattleMatch.vue';
import BattleResult from '../components/BattleResult.vue';
import { useGoodnessMatch } from '../modes/goodness/useGoodnessMatch.js';
import { useFreeGoodnessMatch } from '../modes/freeGoodness/useFreeGoodnessMatch.js';


const definitions = new Map();


export function registerBattleMode(definition) {
  const key = String(definition?.key || '').trim().toLowerCase();
  if (!key) throw new TypeError('Battle mode requires a key.');
  if (definitions.has(key)) throw new Error(`Battle mode already registered: ${key}`);
  definitions.set(key, Object.freeze({ ...definition, key }));
}


export function getBattleModeDefinition(modeKey = 'goodness') {
  const key = String(modeKey || 'goodness').trim().toLowerCase();
  return definitions.get(key) || definitions.get('goodness');
}


export function listBattleModeDefinitions() {
  return Object.freeze(Array.from(definitions.values()));
}


registerBattleMode({
  key: 'goodness',
  version: 1,
  labelKey: 'battle.modes.goodness.name',
  shortLabelKey: 'battle.modes.goodness.shortName',
  summaryKey: 'battle.modes.goodness.summary',
  ruleKeys: Object.freeze([
    'battle.modes.goodness.rules.sharedRoute',
    'battle.modes.goodness.rules.moveCorrection',
    'battle.modes.goodness.rules.scoring',
    'battle.modes.goodness.rules.certainty',
    'battle.modes.goodness.rules.timeout',
    'battle.modes.goodness.rules.spectating',
  ]),
  HallView: BattleHall,
  MatchView: BattleMatch,
  ResultView: BattleResult,
  createSession: useGoodnessMatch,
});

registerBattleMode({
  key: 'free_goodness',
  version: 2,
  labelKey: 'battle.modes.freeGoodness.name',
  shortLabelKey: 'battle.modes.freeGoodness.shortName',
  summaryKey: 'battle.modes.freeGoodness.summary',
  ruleKeys: Object.freeze([
    'battle.modes.freeGoodness.rules.independentBoards',
    'battle.modes.freeGoodness.rules.riskCorrection',
    'battle.modes.freeGoodness.rules.safeSpawns',
    'battle.modes.freeGoodness.rules.scoring',
    'battle.modes.freeGoodness.rules.timeout',
    'battle.modes.freeGoodness.rules.billing',
  ]),
  HallView: BattleHall,
  MatchView: BattleMatch,
  ResultView: BattleResult,
  createSession: useFreeGoodnessMatch,
});
