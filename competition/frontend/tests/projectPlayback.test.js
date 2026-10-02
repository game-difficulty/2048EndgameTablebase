import test from 'node:test';
import assert from 'node:assert/strict';
import { ProjectPlayback } from '../../shared/projectPlayback.mjs';
import { receivedProjectView } from '../../shared/projectStateOrder.mjs';
import { BOARD_QUAKE_DURATION, projectAnimationHoldMs } from '../src/projects/boardMotion.js';

const frame = sequence => ({ sequence, payload: { board: [[sequence]], elapsed_ms: sequence * 80, last_transition: { kind: 'move' } } });
const view = (sequence, frames = [], frame_start = 1) => ({ generation: 1, ...frame(sequence), frames, frame_start });
function playback() {
  const timers = new Map(), seen = []; let serial = 0;
  const player = new ProjectPlayback(value => seen.push(value?.sequence), {
    schedule(fn) { timers.set(++serial, fn); return serial; }, cancel(id) { timers.delete(id); },
  });
  const drain = () => { while (timers.size) { const [id, fn] = timers.entries().next().value; timers.delete(id); fn(); } };
  return { player, seen, drain };
}

test('a burst plays every frame in order, including the final frame', () => {
  const { player, seen, drain } = playback();
  player.receive(view(0), 'A:white');
  player.receive(view(50, Array.from({ length: 50 }, (_, i) => frame(i + 1))), 'A:white');
  drain();
  assert.deepEqual(seen, Array.from({ length: 51 }, (_, i) => i));
});

test('out-of-order batches fill gaps and duplicates never replay a move', () => {
  const { player, seen, drain } = playback();
  player.receive(view(0), 'A:white');
  let received = view(4, [frame(3), frame(4)]);
  player.receive(received, 'A:white');
  assert.deepEqual(seen, [0]);
  received = receivedProjectView(received, view(2, [frame(1), frame(2)]));
  player.receive(received, 'A:white'); drain();
  player.receive(view(2, [frame(1), frame(2)]), 'A:white'); drain();
  assert.deepEqual(seen, [0, 1, 2, 3, 4]);
});

test('short disconnect catches up; lost history and a new game rebase', () => {
  const { player, seen, drain } = playback();
  player.receive(view(20), 'A:white');
  player.receive(view(25, [21, 22, 23, 24, 25].map(frame)), 'A:white'); drain();
  assert.deepEqual(seen, [20, 21, 22, 23, 24, 25]);
  player.receive(view(200, [199, 200].map(frame), 199), 'A:white'); drain();
  assert.equal(player.view.payload.last_transition.kind, 'restore');
  player.receive(view(2, [1, 2].map(frame)), 'B:white'); drain();
  assert.deepEqual(seen.slice(-2), [200, 2]);
});

test('a history gap hidden behind valid queued frames cannot deadlock', () => {
  const { player, seen, drain } = playback();
  player.receive(view(0), 'A');
  const first = view(2, [frame(1), frame(2)]);
  player.receive(first, 'A');
  player.receive(receivedProjectView(first, view(5, [frame(5)], 5)), 'A');
  drain();
  assert.deepEqual(seen, [0, 1, 2, 5]);
  assert.equal(player.pending.size, 0);
});

test('a missing step has a bounded timeout even if no more packets arrive', () => {
  const { player, seen, drain } = playback();
  player.receive(view(0), 'A');
  player.receive(view(5, [frame(5)], 1), 'A');
  drain();
  assert.deepEqual(seen, [0, 5]);
  assert.equal(player.timer, null);
});

test('a suspended viewer cannot accumulate an unbounded playback queue', () => {
  const { player, seen, drain } = playback();
  player.receive(view(0), 'A');
  player.receive(view(129, Array.from({length:129},(_,i)=>frame(i+1))), 'A');
  drain();
  assert.deepEqual(seen,[0,129]);
  assert.equal(player.pending.size,0);
});

function timedPlayback() {
  const timers=new Map(), seen=[], pending=[];let at=0, serial=0;
  const player=new ProjectPlayback(v=>seen.push({sequence:v?.sequence,at}),{
    now:()=>at,
    schedule(fn,delay){timers.set(++serial,{fn,at:at+delay});return serial;},
    cancel:id=>timers.delete(id),onPending:value=>pending.push(value),
  });
  function tick(ms) {
    const end=at+ms;
    while(timers.size){
      const [id,timer]=[...timers].sort((a,b)=>a[1].at-b[1].at)[0];
      if(timer.at>end)break;
      timers.delete(id);at=timer.at;timer.fn();
    }
    at=end;
  }
  return {player,seen,pending,tick};
}
const quakeFrame=sequence=>({...frame(sequence),payload:{...frame(sequence).payload,aftershock:true,last_transition:{kind:'reshape',quake:{cells:[]}}}});

test('catch-up preserves the full quake/shift/spawn timing without slowing ordinary spam',()=>{
  const {player,seen,tick}=timedPlayback();
  player.receive(view(0),'A');
  player.receive(view(4,[quakeFrame(1),frame(2),frame(3),frame(4)]),'A');
  tick(60);assert.equal(seen.at(-1).sequence,1);
  tick(BOARD_QUAKE_DURATION-1);assert.equal(seen.at(-1).sequence,1);
  tick(1);assert.equal(seen.at(-1).sequence,2);
  tick(32);assert.equal(seen.at(-1).sequence,4);
  assert.equal(projectAnimationHoldMs(frame(2).payload),0);
  assert.equal(projectAnimationHoldMs({...quakeFrame(1).payload,last_transition:{kind:'restore'}}),0);
  player.close();
});

test('a final quake remains pending through completion metadata, then releases the result view',()=>{
  const {player,seen,pending,tick}=timedPlayback();
  player.receive(view(0),'A');
  player.receive(view(1,[quakeFrame(1)]),'A');
  tick(60);
  assert.equal(pending.at(-1),true);
  player.receive({...view(1),payload:{...quakeFrame(1).payload,finished:true}},'A');
  tick(BOARD_QUAKE_DURATION-1);assert.equal(pending.at(-1),true);
  tick(1);assert.equal(pending.at(-1),false);
  assert.equal(seen.at(-1).sequence,1);
  player.close();
});
