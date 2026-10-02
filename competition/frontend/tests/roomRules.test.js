import test from 'node:test';
import assert from 'node:assert/strict';
import { ruleSummary, ruleRequest, lineupValid, lineupPolicyDescription } from '../src/roomRules.js';
const base={preset:'custom',team_size:4,series_mode:'all',lineup_policy:'balanced',final_selection:'random',draft_seconds:60,lineup_seconds:180,team_clock_seconds:1800,steps:[{actor:'first',picks:2,bans:1},{actor:'second',picks:2,bans:1}]};
test('custom BP preview validates counts and strips derived fields from requests',()=>{
  assert.deepEqual(ruleSummary(base,7),{games:5,minimum:7,valid:true});
  assert.equal(ruleSummary(base,6).valid,false);
  assert.equal(ruleSummary({...base,lineup_policy:'unique'},7).valid,false);
  assert.equal(ruleSummary({...base,draft_seconds:0},7).valid,false);
  assert.equal(ruleSummary({...base,steps:[{actor:'first',picks:1,bans:0}]},7).valid,false);
  assert.equal('game_count' in ruleRequest({...base,game_count:5}),false);
  assert.equal('steps' in ruleRequest({...base,preset:'bo5'}),false);
});
test('participation permits 5/1/1 while balance requires 3/2/2 for three-player BO7',()=>{
  const rules={team_size:3,game_keys:[...'ABCDEFG'],lineup_policy:'everyone'};
  const concentrated={A:1,B:1,C:1,D:1,E:1,F:2,G:3};
  const balanced={A:1,B:2,C:3,D:1,E:2,F:3,G:1};
  assert.ok(lineupValid(rules,concentrated));
  assert.equal(lineupValid({...rules,lineup_policy:'balanced'},concentrated),false);
  for(const policy of ['everyone','balanced','free']) assert.ok(lineupValid({...rules,lineup_policy:policy},balanced));
  assert.equal(lineupValid(rules,{...concentrated,G:2}),false);
  assert.ok(lineupValid({...rules,lineup_policy:'free'},{...concentrated,G:2}));
  assert.equal(lineupValid({...rules,lineup_policy:'unknown'},balanced),false);
  assert.ok(ruleSummary({...base,lineup_policy:'everyone'},7).valid);
  assert.equal(ruleSummary({...base,lineup_policy:'unknown'},7).valid,false);
});
test('participation uses distinct players when games are fewer than players',()=>{
  const rules={team_size:5,game_keys:[...'ABC'],lineup_policy:'everyone'};
  assert.ok(lineupValid(rules,{A:5,B:2,C:4}));
  assert.equal(lineupValid(rules,{A:5,B:5,C:4}),false);
  assert.ok(lineupValid({team_size:1,game_keys:[...'ABC'],lineup_policy:'everyone'},{A:1,B:1,C:1}));
});
test('every lineup policy has distinct Chinese and English descriptions',()=>{
  for(const lang of ['zh','en']) {
    const descriptions=['unique','everyone','balanced','free'].map(policy=>lineupPolicyDescription(policy,lang));
    assert.ok(descriptions.every(Boolean));
    assert.equal(new Set(descriptions).size,4);
  }
});
test('lineup validation supports variable games, repeated players and balance',()=>{
  const rules={team_size:4,game_keys:['A','B','C','D','E'],lineup_policy:'balanced'};
  assert.ok(lineupValid(rules,{A:1,B:2,C:3,D:4,E:1}));
  assert.equal(lineupValid(rules,{A:1,B:1,C:1,D:1,E:1}),false);
  assert.ok(lineupValid({...rules,lineup_policy:'free'},{A:1,B:1,C:1,D:1,E:1}));
  assert.equal(lineupValid(rules,{A:1,B:2,C:3,D:4,E:5}),false);
  assert.ok(lineupValid(null,{A:1,B:2,C:3}));
});
