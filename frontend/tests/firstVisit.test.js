import test from 'node:test';
import assert from 'node:assert/strict';
import { createFirstVisit } from '../src/human/firstVisit.js';
import { TAB_ROUTES } from '../src/app/siteProfile.js';
test('confirmation deduplicates clicks and ignores stale account responses', async () => {
 let finish, calls=0;
 const flow=createFirstVisit(()=>{calls++;return new Promise(resolve=>finish=resolve);});
 flow.reset(1);const task=flow.confirm();await flow.confirm();assert.equal(calls,1);
 flow.reset(2);finish({first_visit:[]});await task;assert.ok(flow.pending.value);
 flow.reset(null);assert.equal(flow.pending.value,null);
 assert.equal(TAB_ROUTES.contact,'ContactView');
});
test('failed confirmation remains pending and can retry', async () => {
 let calls=0;const flow=createFirstVisit(async()=>{if(!calls++)throw Error();return {first_visit:[]};});
 flow.reset(1);await flow.confirm();assert.ok(flow.error.value);assert.ok(flow.pending.value);
 await flow.confirm();assert.equal(flow.pending.value,null);assert.equal(flow.error.value,'');
});
