import test from 'node:test';
import assert from 'node:assert/strict';
import { participantName,showEndNotice } from '../src/live/content/participants.js';
import { emptyMultiState,receiveMultiJson } from '../src/live/content/multiLiveState.js';
import { boardPipFrame } from '../src/live/content/boardPipFrame.js';
test('names remain stable by seat; ended notice expires without a refresh replay',()=>{
  assert.deepEqual([0,1,2].map(lane=>participantName({lane})),['Lume','Clari','Vero']);
  assert.equal(showEndNotice({ended_at:100},104999),true);
  assert.equal(showEndNotice({ended_at:100},105000),false);
  assert.equal(showEndNotice({ended_at:100},500000),false);
});
test('batch events consume the shared stream sequence and are idempotent',()=>{
  const state={...emptyMultiState(),epoch:'room',seq:3};
  const event={type:'batch',stream_epoch:'room',content_seq:4,batch:{id:'batch',phase:'cooldown'}};
  const next=receiveMultiJson(state,event);
  assert.equal(next.batch.phase,'cooldown');assert.equal(next.seq,4);
  assert.strictEqual(receiveMultiJson(next,event),next);
  assert.throws(()=>receiveMultiJson(state,{...event,content_seq:5}));
});
test('PiP repaints when a death notice expires even if no moves arrive',()=>{
  const config={slots:[{lane:0,name:'Lume',status:'ended',run:{run_id:'run',seq:10,ended_at:100}}],state:'live',lang:'en'};
  assert.notEqual(boardPipFrame({...config,now:104000}).key,boardPipFrame({...config,now:105000}).key);
});
