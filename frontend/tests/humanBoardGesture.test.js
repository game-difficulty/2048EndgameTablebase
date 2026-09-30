import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { boardSwipeDirection } from '../src/components/boardPointerGesture.js';
const source = readFileSync(new URL('../src/human/HumanBoard.vue', import.meta.url), 'utf8');
function harness(overrides = {}) {
  const calls = [];
  const props = { editable: false, swipeSensitivity: 100, touchButton: 0, ...overrides };
  const ctx = vm.createContext({ props, boardSwipeDirection, emit: (...args) => calls.push(args) });
  vm.runInContext('let pointer = null;\n' + source.slice(source.indexOf('function down(e)'), source.indexOf('</script>')), ctx);
  const target = { getBoundingClientRect: () => ({width:350,height:350}), setPointerCapture() {}, hasPointerCapture:()=>true, releasePointerCapture() {} };
  const event = (x=100,y=100,extra={}) => ({clientX:x,clientY:y,pointerId:1,pointerType:'touch',isPrimary:true,button:0,currentTarget:target,
    target:{closest: selector=>selector==='[data-cell]' ? {dataset:{cell:'3'}} : null},preventDefault(){},...extra});
  return {ctx,calls,event};
}
test('board wires immediate motion and cancellation handlers', () => {
  for (const binding of ['@pointermove="move"','@pointercancel="cancel"','@lostpointercapture="cancel"']) assert.ok(source.includes(binding));
});
test('crossing the threshold moves before release, once per gesture even after reversal', () => {
  for (const [x,y,direction] of [[116,100,1],[84,100,3],[100,84,0],[100,116,2]]) {
    const {ctx,calls,event}=harness();ctx.down(event());ctx.move(event(110,100));assert.equal(calls.length,0);
    ctx.move(event(x,y));assert.deepEqual(calls,[['move',direction]]);
    ctx.move(event(200,200));ctx.up(event(20,20));assert.deepEqual(calls,[['move',direction]]);
    ctx.down(event());ctx.move(event(x,y));assert.equal(calls.length,2);
  }
});
test('gaps accept swipes but never place a tile; overlay buttons never start a swipe', () => {
  const {ctx,calls,event}=harness({editable:true});
  const gap={target:{closest:()=>null}};
  ctx.down(event(100,100,gap));ctx.move(event(120,100));ctx.up(event(120,100));assert.deepEqual(calls,[['move',1]]);
  ctx.down(event(100,100,gap));ctx.up(event());assert.equal(calls.length,1);
  ctx.down(event(100,100,{target:{closest:s=>s==='.board-overlay'?{}:null}}));ctx.move(event(200,100));ctx.up(event(200,100));assert.equal(calls.length,1);
});
test('practice taps and right-click placement survive without a placement after swiping', () => {
  const {ctx,calls,event}=harness({editable:true,touchButton:2});
  ctx.down(event());ctx.up(event(103,104));assert.deepEqual(calls,[['cell',3,2]]);
  ctx.down(event());ctx.move(event(120,100));ctx.up(event(120,100));assert.deepEqual(calls,[['cell',3,2],['move',1]]);
  ctx.down(event(100,100,{pointerType:'mouse',button:2}));assert.deepEqual(calls.at(-1),['cell',3,2]);
});
test('cancellation and secondary pointers cannot finish or replace the current gesture', () => {
  const {ctx,calls,event}=harness({editable:true});
  ctx.down(event());ctx.down(event(200,200,{pointerId:2,isPrimary:false}));
  ctx.cancel(event(200,200,{pointerId:2}));ctx.move(event(200,200,{pointerId:2}));assert.equal(calls.length,0);
  ctx.cancel(event());ctx.up(event(150,100));assert.equal(calls.length,0);
  ctx.down(event());ctx.up(event(120,100));assert.deepEqual(calls,[['move',1]]);
});
test('existing sensitivity thresholds remain unchanged', () => {
  const {ctx,calls,event}=harness({swipeSensitivity:50});ctx.down(event());ctx.move(event(120,100));assert.equal(calls.length,0);
  ctx.move(event(132,100));assert.deepEqual(calls,[['move',1]]);
});
