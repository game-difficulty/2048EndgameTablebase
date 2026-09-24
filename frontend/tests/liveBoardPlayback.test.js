import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { LiveBoardPlayback, LIVE_PLAYBACK_MAX_DELAY, LIVE_PLAYBACK_MAX_FRAMES } from '../src/live/content/liveBoardPlayback.js';
import { emptyMultiState, receiveMultiJson, receiveMultiBatch, syncMultiFrames } from '../src/live/content/multiLiveState.js';
import { boardFrameRenderMode } from '../src/components/boardFrame.js';

const fixture=JSON.parse(readFileSync(new URL('./fixtures/live-multi.json',import.meta.url)));
function decoded() {
  const initial=receiveMultiJson(emptyMultiState(),fixture.snapshot), transitions=[];
  const next=receiveMultiBatch(initial,new Uint8Array(fixture.packet).buffer,transitions);
  return {initial,next,transitions};
}

test('batched moves remain continuous for the main board and drain without another network message',()=>{
  const {initial,next,transitions}=decoded();
  const player=new LiveBoardPlayback(initial.frames);
  const laneMoves=transitions.filter(t=>t.lane===0);
  assert.ok(laneMoves.length>1);
  player.enqueue(laneMoves,0);
  let board=initial.frames[0].toBoard;
  laneMoves.forEach(({frame},i)=>{
    const painted=player.paint(i*16,'focus',0)[0];
    assert.strictEqual(painted,frame);
    assert.equal(boardFrameRenderMode(board,painted),'animate');
    board=painted.toBoard;
  });
  assert.equal(player.pending,false);
  assert.deepEqual(board,next.slots[0].run.board);
  assert.deepEqual(initial.slots[0].run.board,fixture.snapshot.lanes[0].run.board);
});

test('equal boards share the same cadence while focus previews drain on their own cadence',()=>{
  const {initial,transitions}=decoded();
  const player=new LiveBoardPlayback(initial.frames);
  player.enqueue(transitions,0);
  const first=player.paint(0,'equal',0);
  assert.deepEqual(player.paint(16,'equal',0),first);
  const second=player.paint(34,'equal',0);
  for(let lane=0;lane<3;lane++)assert.notEqual(second[lane].revision,first[lane].revision);
  player.sync(initial.frames);player.enqueue(transitions,0);
  const focus=player.paint(0,'focus',0);
  const main=player.paint(17,'focus',0);
  assert.notEqual(main[0].revision,focus[0].revision);
  assert.strictEqual(main[1],focus[1]);assert.strictEqual(main[2],focus[2]);
  assert.equal(player.pending,true);
  const previews=player.paint(68,'focus',0);
  assert.notEqual(previews[1].revision,focus[1].revision);
});

test('overdue or excessive buffering jumps forward and never accumulates a long replay',()=>{
  const {initial,next,transitions}=decoded();
  const player=new LiveBoardPlayback(initial.frames);
  player.enqueue(transitions,0);
  const frames=player.paint(LIVE_PLAYBACK_MAX_DELAY+1,'equal',0);
  assert.equal(player.pending,false);
  frames.forEach((frame,lane)=>{assert.equal(frame.kind,'snapshot');assert.deepEqual(frame.toBoard,next.slots[lane].run.board);});
  for(let i=0;i<1000;i++)player.enqueue([transitions[0]],i);
  assert.ok(player.queues[0].length<=LIVE_PLAYBACK_MAX_FRAMES);
});

test('snapshot, layout switch and single-lane termination discard only the intended buffered moves',()=>{
  const {initial,next,transitions}=decoded();
  const player=new LiveBoardPlayback(initial.frames);
  player.enqueue(transitions,0);
  const latest=syncMultiFrames(next).frames;
  player.sync(latest,[1]);
  assert.equal(player.queues[1].length,0);
  assert.ok(player.queues[0].length>0 && player.queues[2].length>0);
  assert.strictEqual(player.frames[1],latest[1]);
  player.sync(latest);
  assert.equal(player.pending,false);
  assert.deepEqual(player.paint(200,'focus',2),latest);
});

test('malformed packets cannot leak any playback frames; duplicate packets do not replay',()=>{
  const {initial,next}=decoded();
  const transitions=[];
  const corrupt=new Uint8Array([...fixture.packet,0,128]).buffer;
  assert.throws(()=>receiveMultiBatch(initial,corrupt,transitions),/snapshot_required/);
  assert.deepEqual(transitions,[]);
  receiveMultiBatch(next,new Uint8Array(fixture.packet).buffer,transitions);
  assert.deepEqual(transitions,[]);
});
