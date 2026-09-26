import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { LiveBoardPlayback, LIVE_PLAYBACK_BASE_INTERVAL } from '../src/live/content/liveBoardPlayback.js';
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
    const painted=player.paint(i*LIVE_PLAYBACK_BASE_INTERVAL,'focus',0)[0];
    assert.strictEqual(painted,frame);
    assert.equal(boardFrameRenderMode(board,painted),'animate');
    board=painted.toBoard;
  });
  assert.equal(player.pending,false);
  assert.deepEqual(board,next.slots[0].run.board);
  assert.deepEqual(initial.slots[0].run.board,fixture.snapshot.lanes[0].run.board);
});

test('all layouts and lanes consume the same queue cadence',()=>{
  const {initial,transitions}=decoded();
  const focus=new LiveBoardPlayback(initial.frames), equal=new LiveBoardPlayback(initial.frames);
  focus.enqueue(transitions,0);equal.enqueue(transitions,0);
  for(let now=0;now<1000;now+=4) {
    assert.deepEqual(focus.paint(now,'focus',0),equal.paint(now,'equal',0));
    assert.deepEqual(focus.queues.map(q=>q.length),equal.queues.map(q=>q.length));
  }
  assert.equal(focus.pending,false);
});

test('long stalls and more than twelve moves preserve every queued frame',()=>{
  const {initial,transitions}=decoded();
  const player=new LiveBoardPlayback(initial.frames);
  const moves=Array.from({length:50},(_,i)=>({lane:0,frame:{...transitions[0].frame,revision:`ordered:${i}`}}));
  player.enqueue(moves,0);
  assert.equal(player.queues[0].length,50);
  let now=5000,previous=initial.frames[0];const seen=[];
  while(player.pending) {
    const frame=player.paint(now)[0];
    if(frame!==previous){seen.push(frame.revision);previous=frame;}
    now+=player.pending?player.nextDelay(now)+0.001:0;
    assert.ok(now<10000);
  }
  assert.deepEqual(seen,moves.map(m=>m.frame.revision));
  assert.equal(previous.kind,'move');
});

test('backlog accelerates gradually and returns gradually to normal cadence',()=>{
  const {initial,transitions}=decoded();const player=new LiveBoardPlayback(initial.frames);
  player.enqueue(Array.from({length:30},()=>transitions[0]));
  let now=0;player.paint(now);
  const first=player.intervals[0];
  assert.ok(first<40 && first>20);
  while(player.queues[0].length>3){now+=player.nextDelay(now)+0.001;player.paint(now);}
  const fast=player.intervals[0];
  now+=player.nextDelay(now)+0.001;player.paint(now);
  assert.ok(player.intervals[0]>fast && player.intervals[0]<40);
});

test('three fast streams release in order without a growing backlog',()=>{
  const {initial,transitions}=decoded();const player=new LiveBoardPlayback(initial.frames);
  let produced=0,consumed=0,maxQueue=0,previous=[...initial.frames];
  for(let now=0;now<10000;now++) {
    if(now<8000 && now%18===0) {
      player.enqueue([0,1,2].map(lane=>({lane,frame:{...transitions[0].frame,revision:`${lane}:${produced}`}})));
      produced++;
    }
    const frames=player.paint(now);
    for(let lane=0;lane<3;lane++)if(frames[lane]!==previous[lane]){consumed++;previous[lane]=frames[lane];}
    maxQueue=Math.max(maxQueue,...player.queues.map(q=>q.length));
  }
  assert.equal(consumed,produced*3);
  assert.equal(player.pending,false);
  assert.ok(maxQueue<20,`backlog ${maxQueue}`);
});

test('explicit snapshot and lane replacement reset only the intended queues',()=>{
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
