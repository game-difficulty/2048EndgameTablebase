import test from 'node:test';
import assert from 'node:assert/strict';
import { reactive } from 'vue';
import { initialState } from '../../../frontend/src/human/engine.js';
import { TimeAttackRuntime, targetReached, attemptStatus, VARIANTS } from '../src/timeAttackRuntime.js';

function runtime(variant='4x4', target=2048){
  const state=initialState('run',variant,'00000000000000000000000000000001');
  return new TimeAttackRuntime(reactive({id:'run',state,status:'playing'}),{variant,target_kind:'tile',target_value:target});
}
test('all four boards produce verified-format events and clone reactive snapshots',()=>{
  for(const variant of Object.keys(VARIANTS)){
    const run=runtime(variant);
    assert.equal(run.state.board.length,VARIANTS[variant][0]*VARIANTS[variant][1]);
    assert.ok([0,1,2,3].some(d=>run.move(d,100)));
    const packet=run.batch('first-command');
    assert.equal(packet.base_sequence,0);
    assert.equal(packet.events[0][1],100);
    run.acknowledge(packet);
    assert.equal(run.pending.length,0);
  }
});
test('in-flight acknowledgements preserve newer queued moves and cannot rewind',()=>{
  const run=runtime();
  [0,1,2,3].some(d=>run.move(d,100));
  const first=run.batch('first-command');
  [0,1,2,3].some(d=>run.move(d,200));
  const before=run.state.seq;
  run.acknowledge(first);
  assert.equal(run.state.seq,before);
  assert.equal(run.batch('next-command').base_sequence,1);
  assert.throws(()=>run.acknowledge(first));
});
test('target semantics are independent of room completion',()=>{
  assert.equal(targetReached({target_kind:'tile',target_value:8},[16,0]),true);
  assert.equal(targetReached({target_kind:'board_sum',target_value:10},[8,4]),false);
  assert.equal(targetReached({target_kind:'board_sum',target_value:10},[8,2]),true);
  assert.equal(attemptStatus({variant:'2x4',target_kind:'board_sum',target_value:10},{board:[8,4,0,0,0,0,0,0]}),'overshot');
});
test('finishing stops this attempt, constructing a new one resets only its local state',()=>{
  const run=runtime('4x4',8);
  for(let i=0;i<100 && run.status==='playing';i++)[0,1,2,3].some(d=>run.move(d,(i+1)*100));
  assert.equal(run.status,'reached');
  assert.equal(run.move(0,99999),false);
  assert.equal(runtime().state.seq,0);
});
