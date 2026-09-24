import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { emptyMultiState, receiveMultiJson, receiveMultiBatch } from '../src/live/content/multiLiveState.js';
import { chooseMainLane, followMain, selectMain, changeLayout, changeMainMode, leaderLanes } from '../src/live/content/mainRunSelection.js';

const snapshot = () => ({ type: 'snapshot', stream_epoch: 'epoch-1', content_seq: 0, sources: { 0: 'AI' },
  lanes: [0,1,2].map(lane => ({ lane, generation: 1, status: 'running', run: {
    run_id: `run-${lane}`, seq: 0, board: [2,2,...Array(14).fill(0)], score: lane * 4, elapsed_ms: 0, source_id: 0,
  } })) });
const batch = (seq, records) => { const data = new Uint8Array([0x21,0,0,0,0,...records]);
  new DataView(data.buffer).setUint32(1, seq, true); return data.buffer; };

test('Python publisher fixtures reproduce all three boards through varint boundaries', () => {
  const fixture = JSON.parse(readFileSync(new URL('./fixtures/live-multi.json', import.meta.url)));
  const state = receiveMultiBatch(receiveMultiJson(emptyMultiState(), fixture.snapshot), new Uint8Array(fixture.packet).buffer);
  assert.deepEqual(state.slots.map(({run}) => ({ board:run.board, score:run.score, seq:run.seq, elapsed_ms:run.elapsed_ms })), fixture.expected);
});

test('three-lane compact decoding computes score, timing and spawn without score packets', () => {
  let state = receiveMultiJson(emptyMultiState(), snapshot());
  // Move left, spawn at cell 1, value=2, elapsed=15ms.
  state = receiveMultiBatch(state, batch(1, [28,30,29,30,30,30]));
  assert.deepEqual(state.slots.map(s => s.run.seq), [1,1,1]);
  assert.deepEqual(state.slots.map(s => s.run.score), [4,8,12]);
  assert.deepEqual(state.slots[0].run.board.slice(0,3), [4,2,0]);
  assert.equal(state.seq, 3);
  assert.equal(state.slots[2].run.elapsed_ms, 15);
});

test('snapshot watermark skips covered batch prefix; malformed suffix never partially commits', () => {
  let initial = receiveMultiJson(emptyMultiState(), snapshot());
  const first = receiveMultiBatch(initial, batch(1,[28,30]));
  const result = receiveMultiBatch(first, batch(1,[28,30,29,30]));
  assert.deepEqual(result.slots.map(s => s.run.seq), [1,1,0]);
  assert.throws(() => receiveMultiBatch(initial,batch(1,[28,30,0,128])), /snapshot_required/);
  assert.equal(initial.slots[0].run.seq, 0);
  assert.throws(() => receiveMultiBatch(initial,batch(2,[28,30])), /snapshot_required/);
  assert.throws(() => receiveMultiBatch(initial,batch(1,[7,0])), /snapshot_required/);
});

test('dictionary/source records are ordered content events, not game moves', () => {
  let state = receiveMultiJson(emptyMultiState(), snapshot());
  state = receiveMultiJson(state,{type:'dictionary',stream_epoch:'epoch-1',content_seq:1,sources:{1:'free10_1024'}});
  state = receiveMultiBatch(state,batch(2,[3,2,1,30,30]));
  assert.equal(state.slots[2].run.source, 'free10_1024');
  assert.equal(state.slots[2].run.seq, 1);
  assert.equal(state.seq, 3);
  assert.throws(() => receiveMultiJson(state,{type:'lane_status',stream_epoch:'old',content_seq:4,lane:0,generation:1,status:'running'}));
});

test('leader badges include finished players, while automatic viewing prefers a surviving player', () => {
  const slots = snapshot().lanes;
  assert.equal(chooseMainLane(slots,0,true),2);
  slots[1].run.score = 8;
  assert.equal(chooseMainLane(slots,1),1);
  assert.deepEqual(leaderLanes(slots),[1,2]);
  slots[1].status = 'recovering'; slots[2].run.ended_at = 1;
  assert.equal(chooseMainLane(slots,2),1);
});

test('equal layout freezes selection, preserves manual lane and returns to latest automatic leader', () => {
  const slots = snapshot().lanes;
  let view = { layout:'focus',mainMode:'auto',selectedLane:0 };
  view = followMain(view,slots,true); assert.equal(view.selectedLane,2);
  view = changeLayout(view,'equal',slots);
  slots[0].run.score = 100;
  assert.equal(followMain(view,slots).selectedLane,2);
  view = changeLayout(view,'focus',slots); assert.equal(view.selectedLane,0);
  view = selectMain(view,1); assert.equal(view.mainMode,'manual');
  view = changeLayout(view,'equal',slots);
  slots[1].run.run_id = 'next-game'; slots[1].run.score = 0;
  view = changeLayout(view,'focus',slots); assert.equal(view.selectedLane,1);
  view = changeMainMode(view,'auto',slots); assert.equal(view.selectedLane,0);
  assert.deepEqual(selectMain({...view,layout:'equal'},2),{layout:'focus',mainMode:'manual',selectedLane:2});
});

test('automatic viewing waits for a 1000-point lead and resists near-score reversals',()=>{
  const slots=snapshot().lanes;
  slots[0].run.score=10000;slots[1].run.score=10996;slots[2].run.score=10200;
  assert.equal(chooseMainLane(slots,0),0);
  slots[1].run.score=11000;
  assert.equal(chooseMainLane(slots,0),1);
  slots[0].run.score=11004;
  assert.equal(chooseMainLane(slots,1),1);
  slots[0].run.score=11996;
  assert.equal(chooseMainLane(slots,1),1);
  slots[0].run.score=12000;
  assert.equal(chooseMainLane(slots,1),0);
});

test('death bypasses the score margin, never returns to a dead player, and final results settle stably',()=>{
  const slots=snapshot().lanes;
  slots[0].run.score=50000;slots[1].run.score=20000;slots[2].run.score=19900;
  slots[0].run.ended_at=123;
  let selected=chooseMainLane(slots,0);assert.equal(selected,1);
  assert.equal(chooseMainLane(slots,selected),1);
  slots[1].status='ended';selected=chooseMainLane(slots,selected);assert.equal(selected,2);
  slots[2].run.ended_at=124;
  selected=chooseMainLane(slots,selected);assert.equal(selected,0);
  assert.equal(chooseMainLane(slots,selected),0);
});

test('all-ended ties retain selection and recovering is not death',()=>{
  const slots=snapshot().lanes;
  slots.forEach(s=>{s.run.score=100;s.status='ended';});
  assert.equal(chooseMainLane(slots,1),1);
  slots.forEach(s=>s.status='running');slots[0].status='recovering';slots[1].run.score=500;
  assert.equal(chooseMainLane(slots,0),0);
});

test('manual and equal modes never auto-switch; explicitly returning to auto selects the best immediately',()=>{
  const slots=snapshot().lanes;
  let view={layout:'focus',mainMode:'manual',selectedLane:0};
  slots[0].run.ended_at=1;
  assert.equal(followMain(view,slots).selectedLane,0);
  view=changeMainMode(view,'auto',slots);assert.equal(view.selectedLane,2);
  view=changeLayout(view,'equal',slots);slots[1].run.score=4000;
  assert.equal(followMain(view,slots).selectedLane,2);
  assert.equal(changeLayout(view,'focus',slots).selectedLane,1);
  slots[0].run.ended_at=null;slots[0].run.score=4004;
  assert.equal(changeMainMode({...view,layout:'focus',mainMode:'manual',selectedLane:1},'auto',slots).selectedLane,0);
});

test('empty or missing current lanes have a stable fallback',()=>{
  assert.equal(chooseMainLane([],2),2);
  assert.equal(chooseMainLane(snapshot().lanes.slice(1),0),2);
});
