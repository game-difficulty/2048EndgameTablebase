import test from 'node:test';
import assert from 'node:assert/strict';
import { choosePipMode, FramePump, moveSurface } from '../src/live/pip/roomPip.js';
import { boardPipFrame } from '../src/live/content/boardPipFrame.js';

test('document PiP works without a game renderer; fallback requires an explicit adapter', () => {
  assert.equal(choosePipMode('auto',{document:true,video:false,renderer:false}),'document');
  assert.equal(choosePipMode('auto',{document:false,video:true,renderer:true}),'video');
  assert.equal(choosePipMode('video',{document:true,video:true,renderer:false}),null);
  assert.equal(choosePipMode('document',{document:false,video:true,renderer:true}),null);
});
test('frame pump coalesces latest content and updates without requestAnimationFrame', () => {
  let clock=0,key='chess-1'; const drawn=[];
  const pump=new FramePump(()=>({key,width:960,height:540}),frame=>drawn.push(frame.key),()=>clock);
  pump.tick(); key='chess-2';clock=40;pump.tick();key='chess-3';clock=100;pump.tick();
  assert.deepEqual(drawn,['chess-1','chess-3']);
  clock=900;pump.tick();assert.equal(drawn.length,2);
  clock=1100;pump.tick();assert.equal(drawn.length,3);
});
test('surface returns to its exact slot once, preserving the same component node', () => {
  const events=[],surface={ownerDocument:{createComment:()=>marker},before:node=>events.push(['marker',node])};
  const marker={parentNode:{},replaceWith:node=>events.push(['restore',node])};
  const undo=moveSurface(surface,{append:node=>events.push(['move',node])});
  undo();undo();assert.equal(events.length,3);assert.equal(events[1][1],surface);assert.equal(events[2][1],surface);
});
test('classic adapters keep a fixed 16:9 frame and track every lane, selection and status', () => {
  const slots=[0,1,2].map(lane=>({lane,status:'running',run:{seq:1,run_id:String(lane),board:Array(16).fill(0)}}));
  const props={slots,state:'live',lang:'zh',layout:'focus',selectedLane:0};
  const a=boardPipFrame(props); slots[2].run.seq++;
  const b=boardPipFrame(props), c=boardPipFrame({...props,layout:'equal'});
  assert.notEqual(a.key,b.key);assert.notEqual(b.key,c.key);
  assert.equal(a.width/a.height,16/9);assert.equal(c.width,a.width);assert.equal(c.height,a.height);
});
