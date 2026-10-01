import test from 'node:test';
import assert from 'node:assert/strict';
import { projectRuleMetrics } from '../../shared/projectMetrics.mjs';

test('seal rotation uses authoritative remaining moves, not total moves', () => {
  for (const [moves, next, warning] of [[0,100,false],[99,1,true],[100,100,false],[199,1,true],[200,100,false]]) {
    const [metric]=projectRuleMetrics({payload:{move_count:moves,next_seal_in:next}},'en');
    assert.equal(metric.value,`${next} moves`);
    assert.equal(metric.warning,warning);
  }
  assert.deepEqual(projectRuleMetrics({payload:{move_count:99}}),[]);
});

test('full load shows numeric tiles against capacity, not empty cells or walls', () => {
  const payload={board:[[2,0,-1,4],[8,16,0,0],[32,64,128,256],[512,1024,2048,4096]]};
  assert.deepEqual(projectRuleMetrics({payload},'en',{project_ref:'practice-full-load-4x4'}),[{key:'capacity',label:'TILES / LIMIT',value:'12 / 12',warning:true}]);
  assert.deepEqual(projectRuleMetrics({payload},'en'),[]);
  assert.equal(projectRuleMetrics({payload:{...payload,tile_limit:20}},'en')[0].value,'12 / 20');
});

test('targets are read from the public match payload, including archived targets', () => {
  assert.equal(projectRuleMetrics({payload:{target_sum:1022}},'en')[0].value,'1,022');
  assert.equal(projectRuleMetrics({payload:{target_tile:64,target_count:10}},'en')[0].value,'10');
  assert.equal(projectRuleMetrics({payload:{target_tile:2048,target_count:1}})[0].label,'2048砖目标');
});

test('cargo opening counter stops after first cargo, with no invented time limit', () => {
  const view={view_protocol:'cargo-transport-v1',payload:{move_count:9,cargo:null}};
  assert.equal(projectRuleMetrics(view)[0].value,'1 步');
  assert.deepEqual(projectRuleMetrics({...view,payload:{move_count:10,cargo:{}}}),[]);
  assert.deepEqual(projectRuleMetrics({...view,payload:{cargo:null}}),[]);
});

test('unrelated and private timers never add speculative rule indicators', () => {
  assert.deepEqual(projectRuleMetrics({payload:{move_count:50,score:1000,fission_timers:[5]}}),[]);
  assert.deepEqual(projectRuleMetrics(null),[]);
});
