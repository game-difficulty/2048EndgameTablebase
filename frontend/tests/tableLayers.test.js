import test from 'node:test';
import assert from 'node:assert/strict';
import { compileTableLayers, tableLayerAvailable } from '../src/features/gamer/engine/tableLayers.js';
import { TableDispatcher, maskLargeTiles } from '../src/features/gamer/engine/tableDispatcher.js';
import { TableAiCache } from '../src/features/gamer/services/tableAiCache.js';
import { simulateMove } from '../src/features/gamer/engine/classicMove.js';

const board = [...'10212253c73fda6e'].map(c => c === '0' ? 0 : 2 ** parseInt(c, 16));
const layers = (n, seed, ranges) => ({ version: 1, nums_adjust: -32768 * n - seed, ranges });

test('reported board needs free12 layer 629 and free11 layer 118', () => {
  const free12 = compileTableLayers(layers(4, 22, [[0, 576]]));
  const free11 = compileTableLayers(layers(5, 20, [[0, 500]]));
  assert.equal(tableLayerAvailable(maskLargeTiles(board, 4), free12), false);
  assert.equal(tableLayerAvailable(maskLargeTiles(board, 4), compileTableLayers(layers(4, 22, [[629, 629]]))), true);
  assert.equal(tableLayerAvailable(maskLargeTiles(board, 5), free11), true);
  assert.equal(tableLayerAvailable(maskLargeTiles(board, 5), compileTableLayers(layers(5, 20, [[118, 118]]))), true);
});

test('holes and inclusive boundaries are respected', () => {
  const coverage = compileTableLayers({ version: 1, nums_adjust: 0, ranges: [[1, 2], [4, 5]] });
  for (const [layer, expected] of [[0,false],[1,true],[2,true],[3,false],[4,true],[5,true],[6,false]]) {
    assert.equal(tableLayerAvailable([layer * 2], coverage), expected);
  }
  assert.equal(tableLayerAvailable([2], compileTableLayers(layers(0, 0, []))), false);
});

test('unknown, malformed and future metadata retain original lookup behavior', () => {
  for (const value of [null, {}, {version:2}, layers(0,0,[[3,1]]), layers(0,0,[[1,3],[2,4]]), layers(0,0,[[0,Infinity]])]) {
    assert.equal(compileTableLayers(value), null);
    assert.equal(tableLayerAvailable(board, compileTableLayers(value)), true);
  }
});

test('missing free12 layer sends no lookup and selects free11 instead', async () => {
  const makeTable = (pattern,target,n,free,seed,ranges) => ({ pattern, target, fullPattern:`${pattern}_${target}`, spawnRate:.1,
    ai:{compatible:true,policy_version:1,large_tiles:n,free_tiles:free,layers:layers(n,seed,ranges)} });
  const tables = [makeTable('free12',2048,4,12,22,[[0,576]]), makeTable('free11',1024,5,11,20,[[0,500]])];
  const dispatcher = new TableDispatcher(tables,.1);
  dispatcher.reset(board);
  const calls=[];
  const choose=()=>dispatcher.choose(async candidate=>{
    calls.push(candidate.table.fullPattern);
    return {results:{left:.874918794},dtype:'uint32'};
  });
  assert.equal(await choose(),'left');
  assert.deepEqual(calls,['free11_1024']);
  // New catalog data restores a previously missing layer without a page reload.
  tables[0].ai.layers=layers(4,22,[[0,700]]);
  dispatcher.setTables(tables,.1);dispatcher.reset(board);calls.length=0;
  await choose();assert.deepEqual(calls,['free12_2048']);
});

test('skipping a missing layer preserves the valid prefetched subscription on the next move', async () => {
  const tables = [
    {pattern:'free12',target:2048,fullPattern:'free12_2048',spawnRate:.1,
      ai:{compatible:true,policy_version:1,large_tiles:4,free_tiles:12,layers:layers(4,22,[[0,576]])}},
    {pattern:'free11',target:1024,fullPattern:'free11_1024',spawnRate:.1,
      ai:{compatible:true,policy_version:1,large_tiles:5,free_tiles:11,layers:layers(5,20,[[0,500]])}},
  ];
  const next = simulateMove(board,'left').board;
  next[next.indexOf(0)] = 2;
  const codes = values => values.map(v=>v?Math.log2(v):0);
  const requests=[];
  const cache=new TableAiCache({transport:{open(body,callbacks){
    const request={body,cancelled:false};requests.push(request);
    queueMicrotask(()=>{
      callbacks.onResult({...body,seq:0,dtype:'uint32',results:{left:.87}});
      callbacks.onResult({...body,seq:1,board_codes:codes(next),dtype:'uint32',results:{left:.88}});
    });
    return {cancel(){request.cancelled=true;},credit(){}};
  },close(){}}});
  try {
    const dispatcher=new TableDispatcher(tables,.1);
    for(const values of [board,next]){
      dispatcher.reset(values);
      assert.equal(await dispatcher.choose(candidate=>cache.lookup({full_pattern:candidate.table.fullPattern,
        catalog_version:'test',board_codes:codes(values),rng_state:[1,2,3,4],difficulty:0,spawn_rate4:.1,random_only:false})), 'left');
    }
    assert.equal(requests.length,1);
    assert.equal(requests[0].body.full_pattern,'free11_1024');
    assert.equal(requests[0].cancelled,false);
  } finally {cache.close();}
});
