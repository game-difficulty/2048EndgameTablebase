import assert from 'node:assert/strict';
import test from 'node:test';
import { TableDispatcher, completePositiveMoves, maskLargeTiles, packedLookupBoard } from '../src/features/gamer/engine/tableDispatcher.js';

const board = code => [...code].map(c => c === '0' ? 0 : 2 ** parseInt(c,16));
const candidate = (pattern='free10',type=1) => ({type,table:{pattern,n:6,target:256,fullPattern:pattern+'_256'}});
const payload = (results,mask=15,dtype='uint32') => ({results,legal_moves_mask:mask,dtype});

test('real free10-256 fixtures accept complete early layer and reject partial layer', async () => {
  const d = new TableDispatcher();
  d.reset(board('101022109830edba'));
  const rates = payload({down:.997717415,left:.997716134,right:.997092993,up:.961990461});
  assert.equal(d.accept(candidate(),rates),'down');
  assert.equal(packedLookupBoard(maskLargeTiles(d.board,6)),'10102210ff30ffff');
  d.reset(board('101202229813edba'));
  assert.equal(d.accept(candidate(),payload({right:.998087728,left:.997920386,down:0,up:0})),null);
  assert.equal(d.cooldowns.size,0);
  d.candidates = () => [candidate(),candidate('fallback')];
  let calls=0;
  assert.equal(await d.choose(async () => {
    calls++;
    return calls===1 ? payload({right:.99,left:.98,down:0,up:0}) : payload({left:.95},1);
  }),'left');
  assert.equal(calls,2);
});

test('coverage ignores illegal directions, requires metadata, and decodes failure-rate types', () => {
  assert.equal(completePositiveMoves(payload({left:.9,right:null},1)),true);
  for (const right of [undefined,null,0,NaN,Infinity,'0.9']) {
    assert.equal(completePositiveMoves(payload({left:.9,right},3)),false);
  }
  assert.equal(completePositiveMoves({results:{left:1}}),false);
  assert.equal(completePositiveMoves(payload({left:1},0)),false);
  assert.equal(completePositiveMoves(payload({left:-.1},1,'1-float32')),true);
  assert.equal(completePositiveMoves(payload({left:-1},1,'1-float32')),false);
});

test('28 generic and 32 free10 boundaries; complete certainty bypasses only small-remainder handoff', () => {
  const d = new TableDispatcher();
  for (const [sum,pattern,expected] of [[26,'ordinary',null],[28,'ordinary','left'],[30,'free10',null],[32,'free10','left']]) {
    const b=board('101022109830edba'); b[1]=sum-22; d.reset(b);
    assert.equal(d.accept(candidate(pattern),payload({left:.9,right:0},3)),expected);
  }
  for (const type of [1,3]) {
    d.reset(board('101022109830edba'));
    assert.equal(d.accept(candidate('ordinary',type),payload({left:1},1)),'left');
    assert.equal(d.cooldowns.size,0);
  }
  assert.equal(d.accept(candidate('ordinary',2),payload({left:1},1)),'AI');
});
