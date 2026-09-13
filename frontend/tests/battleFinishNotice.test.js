import assert from 'node:assert/strict';
import test from 'node:test';
import { battleFinishNotices, createBattleFinishDismissals } from '../src/features/battle/core/battleFinishNotice.js';

const result = (id, goodness, status = 'completed', mode_data = {}) => ({
  actor_key: `u:${id}`, status, goodness_of_fit: goodness, mode_data,
});
const room = (results, final = true, mode = 'goodness') => ({
  room_id: 'room', round: { round_id: 'round', status: final ? 'completed' : 'running' },
  results, mode_key: mode,
});

test('personal finish shows remaining players without prematurely declaring a winner', () => {
  const notices = battleFinishNotices(room([result(1, 1), result(2, .9, 'playing'), result(3, 1, 'disconnected')], false));
  assert.equal(notices['u:1'].title, 'finished');
  assert.equal(notices['u:1'].remaining, 2);
  assert.equal(notices['u:1'].rank, null);
  assert.equal(notices['u:1'].winner, false);
  assert.equal(notices['u:2'], undefined);
  assert.equal(notices['u:3'], undefined);
});
test('final notices use the existing ranking, including ties and solo rounds', () => {
  const notices = battleFinishNotices(room([result(1, .99), result(2, .99), result(3, .9)]));
  assert.equal(notices['u:1'].title, 'jointFirst');
  assert.equal(notices['u:2'].rank, 1);
  assert.equal(notices['u:3'].title, 'placed');
  assert.equal(notices['u:3'].rank, 3);
  assert.equal(battleFinishNotices(room([result(1, .99), result(2, .9)]))['u:1'].title, 'winner');
  const solo = battleFinishNotices(room([result(1, .99)]))['u:1'];
  assert.equal(solo.title, 'solo');
  assert.equal(solo.winner, false);
});
test('timeouts and withdrawals stay unranked regardless of goodness', () => {
  const notices = battleFinishNotices(room([
    result(1, .6), result(2, 1, 'timed_out'), result(3, 1, 'disqualified', {finish_reason:'forfeit'}),
  ]));
  assert.equal(notices['u:1'].title, 'winner');
  assert.equal(notices['u:2'].title, 'timedOut');
  assert.equal(notices['u:2'].rank, null);
  assert.equal(notices['u:3'].title, 'exited');
  assert.equal(notices['u:3'].rank, null);
});
test('free battle does not declare a high-goodness but ineligible player the winner', () => {
  const notices = battleFinishNotices(room([
    result(1, 1), result(2, .8, 'completed', {ranking_eligible:true}),
  ], true, 'free_goodness'));
  assert.equal(notices['u:1'].title, 'unranked');
  assert.equal(notices['u:1'].rank, null);
  assert.equal(notices['u:2'].title, 'winner');
});
test('dismissal persists across remounts, but each round and phase has a separate notice', () => {
  const values = new Map();
  const storage = {getItem:key=>values.get(key),setItem:(key,value)=>values.set(key,value)};
  const state = createBattleFinishDismissals(()=>storage);
  const source = room([result(1, .9)], false);
  const personal = battleFinishNotices(source)['u:1'];
  state.dismiss(personal.key);
  assert.equal(state.has(battleFinishNotices(structuredClone(source))['u:1'].key), true);
  assert.equal(createBattleFinishDismissals(()=>storage).has(personal.key), true);
  source.round.status = 'completed';
  const final = battleFinishNotices(source)['u:1'];
  assert.equal(state.has(final.key), false);
  state.dismiss(final.key);
  source.round.round_id = 'round-2';
  assert.equal(state.has(battleFinishNotices(source)['u:1'].key), false);
});
test('unavailable, full or malformed session storage cannot break finish notices', () => {
  const memory = createBattleFinishDismissals(()=>{throw new Error('blocked')});
  memory.dismiss('key');
  assert.equal(memory.has('key'), true);
  const malformed = createBattleFinishDismissals(()=>({getItem:()=>'{',setItem:()=>{throw new Error('full')}}));
  malformed.dismiss('key');
  assert.equal(malformed.has('key'), true);
  assert.deepEqual(battleFinishNotices(null), {});
});
test('stored dismissals are bounded and scoped to actor identity', () => {
  let saved = '';
  const state = createBattleFinishDismissals(()=>({getItem:()=>null,setItem:(_key,value)=>{saved=value}}));
  for(let index=0; index<100; index+=1) state.dismiss(`key-${index}`);
  assert.equal(JSON.parse(saved).length, 64);
  const notices = battleFinishNotices(room([result(1,.9),result(2,.8)]));
  state.dismiss(notices['u:1'].key);
  assert.equal(state.has(notices['u:2'].key), false);
});
