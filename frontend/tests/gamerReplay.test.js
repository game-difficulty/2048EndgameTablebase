import assert from 'node:assert/strict';
import test from 'node:test';
import { GamerReplay } from '../src/features/gamer/engine/gamerReplay.js';
import { simulateMove } from '../src/features/gamer/engine/classicMove.js';
import { crc32 } from '../src/features/gamer/engine/rankedReplayEncoder.js';
await import('../public/verse-replay/replay-core.js');
const { decodeReplayText, decodeReplayBytes } = globalThis.ReplayCore;
const start = [2,2,...Array(14).fill(0)];
const step = (board,dir,index,value) => { const next=simulateMove(board,dir);next.board[index]=value;return next; };

test('ordinary moves export text and files with exact timing, score and spawns',()=>{
  const replay=new GamerReplay(start,{now:100});
  let state=step(start,'left',15,4);replay.append('left',{index:15,value:4},225);
  const firstScore=state.scoreDelta;
  state=step(state.board,'up',14,2);replay.append('up',{index:14,value:2},1100);
  assert.deepEqual(replay.statistics(),{moves:2,fours:1,rate:.5,partial:false});
  const text=replay.encode();
  for(const decoded of [decodeReplayText(text),decodeReplayBytes(new TextEncoder().encode(text))]){
    assert.equal(decoded.moveCount,2);
    assert.deepEqual([...decoded.getBoardAt(2)],state.board.map(v=>v?Math.log2(v):0));
    assert.equal(decoded.scores[2],firstScore+state.scoreDelta);
    assert.deepEqual(decoded.steps.map(s=>s.spawnValue),[4,2]);
    assert.equal(decoded.knownTimeMs,1000);
  }
});

test('undo removes the abandoned branch and its 4 before recording a new branch',()=>{
  const replay=new GamerReplay(start,{now:0});
  replay.append('left',{index:15,value:4},100);
  replay.undo(200);
  assert.deepEqual(replay.statistics(),{moves:0,fours:0,rate:null,partial:false});
  replay.append('right',{index:0,value:2},250);
  const decoded=decodeReplayText(replay.encode());
  assert.equal(decoded.steps[0].direction,'right');
  assert.equal(decoded.steps[0].spawnValue,2);
  assert.equal(decoded.knownTimeMs,50);
});

test('zero moves and custom starts including 65K use existing RPL1 checkpoints',()=>{
  const board=[32768,32768,65536,0,...Array(12).fill(0)];
  const replay=new GamerReplay(board,{now:0});
  const empty=decodeReplayText(replay.encode());
  assert.equal(empty.moveCount,0);
  assert.deepEqual([...empty.getBoardAt(0)],[15,15,16,0,...Array(12).fill(0)]);
  const state=step(board,'left',15,2);
  replay.append('left',{index:15,value:2},80);
  const decoded=decodeReplayText(replay.encode());
  assert.equal(decoded.scores[1],65536);
  assert.deepEqual([...decoded.getBoardAt(1)],state.board.map(v=>v?Math.log2(v):0));
  assert.equal(decodeReplayText(new GamerReplay(start).encode()).moveCount,0);
});

test('session restoration preserves the recorded history and score',()=>{
  const replay=new GamerReplay(start,{now:0});let board=start,score=0;
  for(let index=0;index<1100;index++){
    const direction=['left','up','right','down'].find(dir=>simulateMove(board,dir).board.some((v,i)=>v!==board[i]));
    if(!direction)break;
    const next=simulateMove(board,direction);const position=next.board.indexOf(0);
    board=next.board;board[position]=index%3===0?4:2;score+=next.scoreDelta;
    replay.append(direction,{index:position,value:board[position]},index*100);
  }
  const restored=GamerReplay.restore(JSON.parse(JSON.stringify(replay.snapshot())),board);
  assert.deepEqual(restored.statistics(),replay.statistics());
  assert.equal(restored.encode(),replay.encode());
  assert.equal(decodeReplayText(restored.encode()).scores[restored.moves.length],score);
});

test('old ranked records recover while stale, malformed and ordinary saves are explicitly partial',()=>{
  const board=step(start,'left',15,4).board;
  const ranked={eligible:true,initialTiles:[[0,0],[1,0]],records:[[1,20],[0,3,15,1,100],[4]]};
  const recovered=GamerReplay.restore(null,board,ranked);
  assert.deepEqual(recovered.statistics(),{moves:1,fours:1,rate:1,partial:false});
  assert.equal(GamerReplay.restore(recovered.snapshot(),start).partial,true);
  assert.equal(GamerReplay.restore(null,board).partial,true);
  assert.equal(GamerReplay.restore(null,board,{...ranked,usedUndo:true}).partial,true);
  assert.equal(GamerReplay.restore(null,board,{...ranked,initialTiles:[null,null]}).partial,true);
  const partial=GamerReplay.restore(null,board);
  assert.deepEqual([...decodeReplayText(partial.encode()).getBoardAt(0)],board.map(v=>v?Math.log2(v):0));
});

test('viewer rejects malformed checkpoints and exports respect its actual file size limit',()=>{
  const initial=new GamerReplay(Array(16).fill(0)).encode();
  let bytes=Uint8Array.from(atob(initial.split('_B64_')[1]),ch=>ch.charCodeAt(0));
  const body=[...bytes.slice(0,7),0,0,...bytes.slice(7,-4)];
  const crc=crc32(body);body.push(crc&255,(crc>>>8)&255,(crc>>>16)&255,crc>>>24);
  assert.throws(()=>decodeReplayText('REPLAY_v1RPL_B64_'+btoa(String.fromCharCode(...body))),/起始局面/);
  const big=new GamerReplay(start);big.moves=Array.from({length:100000},()=>[0,0,0,1000000]);
  assert.throws(()=>big.encode(),/replay_too_large/);
});
