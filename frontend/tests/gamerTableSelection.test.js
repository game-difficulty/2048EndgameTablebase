import assert from 'node:assert/strict';
import test from 'node:test';
import { TableDispatcher } from '../src/features/gamer/engine/tableDispatcher.js';
import { normalizeTableSelection, tableAllowed } from '../src/features/gamer/engine/tableSelection.js';
const table=name=>({pattern:name,fullPattern:name+'_256',target:'256',spawnRate:.1,ai:{compatible:true,policy_version:1,large_tiles:7,free_tiles:3}});

test('default allows newly available tables while explicit empty selection allows none',()=>{
  assert.equal(normalizeTableSelection(undefined),null);
  assert.deepEqual(normalizeTableSelection(['L3_256',null,'L3_256','442_512']),['442_512','L3_256']);
  assert.equal(tableAllowed(table('new'),null),true);
  assert.equal(tableAllowed(table('L3'),[]),false);
});

test('selection is applied before candidate generation or any network lookup',async()=>{
  const tables=[table('L3'),table('442')],dispatcher=new TableDispatcher();
  dispatcher.setTables(tables,.1,['442_256']);
  assert.deepEqual(dispatcher.tables.map(t=>t.fullPattern),['442_256']);
  dispatcher.setTables(tables,.1,[]);
  dispatcher.reset([32768,16384,8192,4096,2048,1024,512,256,128,64,32,16,8,4,2,0]);
  let queries=0;assert.equal(await dispatcher.choose(()=>{queries++;throw Error('Not allowed');}),'AI');
  assert.equal(queries,0);
  dispatcher.setTables(tables,.1,null);
  assert.equal(dispatcher.tables.length,2);
});

test('selected tables still obey existing AI policy and spawn-rate eligibility',()=>{
  const dispatcher=new TableDispatcher();
  dispatcher.setTables([table('L3'),{...table('442'),spawnRate:.2},{...table('444'),ai:null}],.1,['L3_256','442_256','444_256']);
  assert.deepEqual(dispatcher.tables.map(t=>t.fullPattern),['L3_256']);
});
