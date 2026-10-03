import assert from 'node:assert/strict';
import test from 'node:test';
import { compareGoalTargets, fullPatternLabel, isSumTarget, parseGoalTarget, replayPatternFromFilename, sumGoalCompleted } from '../src/utils/goalTarget.js';
import { createPracticeSession, reducePracticeSession } from '../src/features/practice/engine/practiceSession.js';
import { applyTesterLocalMove, createTesterLocalSession, encodeTesterReplay } from '../src/features/tester/engine/testerLocalSession.js';
import { parseRplArrayBuffer } from '../src/features/replay/engine/rplParser.js';
import { aiCompatibleTable } from '../src/features/gamer/engine/tableSelection.js';

test('goal identities, labels, ordering and replay names retain sum tokens', () => {
  assert.deepEqual(['sum-1800','1024','sum-900','512'].sort(compareGoalTargets), ['512','1024','sum-900','sum-1800']);
  assert.equal(fullPatternLabel('3x3_sum-1800', 'zh'), '3x3 · 盘面和 1800');
  for (const value of ['sum-3','sum-16384','sum-901','garbage']) assert.equal(parseGoalTarget(value), null);
  assert.equal(isSumTarget('sum-900'), true);
  for (const [filename, expected] of [['3x3_sum-1800_0.9000.rpl','3x3_sum-1800'], ['2x4_sum-900_stage-1.rpl','2x4_sum-900'], ['L3_256_5_0.9000.rpl','L3_256'], ['3x4free9_512.rpl','3x4free9_512']]) assert.equal(replayPatternFromFilename(filename), expected);
  assert.equal(aiCompatibleTable({pattern:'free10',target:'sum-900',ai:{compatible:true,policy_version:1}}), false);
});
const transition = (tiles, type = 'MOVE_RANDOM') => reducePracticeSession(createPracticeSession({ board:[...tiles, ...new Array(16-tiles.length).fill(0)] }), {type,direction:'down',randomSource:()=>0.5}).state;
test('completion uses an accepted pre-spawn move, not a rounded 100% or spawned sum', () => {
  assert.equal(sumGoalCompleted('sum-900', transition([512,256,128]), 1, 'float64'), false);
  const state = transition([512,256,128,2]);
  assert.equal(sumGoalCompleted('sum-900', state, 1, 'float64'), true);
  assert.equal(sumGoalCompleted('sum-900', state, 0, '1-float64'), true);
  assert.equal(sumGoalCompleted('sum-900', state, .999, 'float64'), false);
  assert.equal(sumGoalCompleted('sum-900', {...state,transition:{kind:'snapshot'}},1,'float64'),false);
  assert.equal(sumGoalCompleted('sum-900', transition([512,256,128,2], 'MOVE_ONLY'),1,'float64'),true);
  assert.equal(sumGoalCompleted('1024', state,1,'float64'), false);
  assert.equal(sumGoalCompleted('sum-900',transition([8192,8192,2]),1,'float64'),false);
});
test('tester retains final move and replay sentinel, and blocks further moves', () => {
  const initial=createTesterLocalSession({board:[512,256,128,2,...new Array(12).fill(0)]});
  const action={target:'sum-900',direction:'down',results:{down:1,left:1},dtype:'float64',randomSource:()=>0.5};
  const moved=applyTesterLocalMove(initial, action);
  assert.equal(moved.accepted,true);
  assert.equal(moved.session.goalCompleted,true);
  assert.equal(moved.session.records.length,1);
  assert.equal(parseRplArrayBuffer(encodeTesterReplay(moved.session)).moveCount,1);
  assert.equal(applyTesterLocalMove(moved.session,action).reason,'goal_completed');
});
