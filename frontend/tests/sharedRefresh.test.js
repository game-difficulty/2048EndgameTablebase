import test from 'node:test';
import assert from 'node:assert/strict';
import {createSharedRefresh} from '../src/live/sharedRefresh.js';
test('socket and foreground share one in-flight and recently finished refresh',async()=>{
  let finish,calls=0,time=0;
  const refresh=createSharedRefresh(()=>{calls++;return new Promise(resolve=>{finish=resolve;});},()=>time);
  const a=refresh(),b=refresh();assert.equal(a,b);await Promise.resolve();assert.equal(calls,1);
  finish({chat:[]});await a;await refresh();assert.equal(calls,1);
  time=2001;const c=refresh();await Promise.resolve();assert.equal(calls,2);finish({chat:[]});await c;
});
test('failed refresh is retriable immediately',async()=>{
  let calls=0;const refresh=createSharedRefresh(async()=>{if(++calls===1)throw Error('offline');return {};});
  await assert.rejects(refresh());await refresh();assert.equal(calls,2);
});
