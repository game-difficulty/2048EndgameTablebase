import assert from 'node:assert/strict';
import test from 'node:test';
import { TableAiCatalog } from '../src/features/gamer/services/tableAiCatalog.js';

test('only the first catalog load blocks; refresh coalesces and preserves usable data on failure', async () => {
  let now=0, calls=0, resolve, reject;
  const catalog=new TableAiCatalog({now:()=>now,load:()=>{
    calls++; return new Promise((yes,no)=>{resolve=yes;reject=no});
  }});
  const first=catalog.get(), second=catalog.get();
  assert.equal(first,second); await Promise.resolve(); assert.equal(calls,1);
  const old={catalogVersion:'v1'}; resolve(old); assert.equal(await first,old);
  now=60001;
  assert.equal(await catalog.get(),old); assert.equal(await catalog.get(),old); assert.equal(calls,2);
  const task=catalog.pending.promise; reject(new Error('offline')); await assert.rejects(task);
  assert.equal(await catalog.get(),old); assert.equal(calls,2);
  now+=5001; assert.equal(await catalog.get(),old);
  const refreshed=catalog.pending.promise; resolve({catalogVersion:'v2'}); await refreshed;
  assert.equal((await catalog.get()).catalogVersion,'v2'); catalog.clear();
});

test('clearing an in-flight catalog aborts and prevents stale results restoring it', async () => {
  let resolve, signal;
  const catalog=new TableAiCatalog({load:(options)=>{signal=options.signal;return new Promise(yes=>{resolve=yes})}});
  const pending=catalog.get(); await Promise.resolve(); catalog.clear();
  assert.equal(signal.aborted,true); resolve({catalogVersion:'old'}); await pending;
  assert.equal(catalog.tables,null);
});
