import test from 'node:test';
import assert from 'node:assert/strict';
import { beijingInput, beijingISO, fixtureStatus, stageGroups } from '../src/fixtureSchedule.js';

test('booking datetime is explicitly Beijing time, independent of device timezone', () => {
  assert.equal(beijingISO('2026-10-04T20:00'), '2026-10-04T12:00:00.000Z');
  assert.equal(beijingInput('2026-10-04T12:00:00Z'), '2026-10-04T20:00');
  assert.equal(beijingInput('2026-10-04T20:00:00+08:00'), '2026-10-04T20:00');
  for (const value of ['2026-02-30T20:00','2026-10-04T25:00','', 'garbage']) assert.equal(beijingISO(value), null);
});

test('fixture status is bilingual and does not leak backend state keys', () => {
  for (const status of ['UNSCHEDULED','SEATING','READY_CHECK','DRAW','DRAFT_STEP','GAME_A_PLAYING','FINISHED','CANCELLED']) {
    for (const lang of ['zh','en']) assert.notEqual(fixtureStatus({status},lang), status);
  }
  assert.equal(fixtureStatus({status:'READY_CHECK',proposed_at:'time'}), '改期待确认');
  assert.equal(fixtureStatus({status:'FINISHED',exception:'both_late'},'en'), 'Neither team ready · 0:0');
});

test('group submissions use stable team IDs, not names or positional guesses', () => {
  const teams = [{id:'pku',name:'PKU'}, {id:'bbb',name:'BBB'}, {id:'fresh',name:'鲜'}];
  assert.deepEqual(stageGroups(teams, {pku:0,bbb:1,fresh:0}, ['A','B']), [
    {name:'A',team_ids:['pku','fresh']}, {name:'B',team_ids:['bbb']},
  ]);
});
