import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
mock.module('../src/services/auth/sessionTokenStore.js', { namedExports: { authHeaders: headers => headers } });
const { upload } = await import('../src/human/client.js');
test('uploads report millisecond-precision first move time in seconds without changing event bytes', async () => {
  const original = globalThis.fetch;
  const calls=[];
  globalThis.fetch=async (url,options)=>{calls.push(options);return {ok:true,json:async()=>({seq:1})};};
  try {
    const run={id:'run',seq:1,firstMoveAt:1790747000123,initialHash:'hash'};
    for(const action of ['monitor','append','reentry','seal'])await upload(run,[[0,0]],'browser','writer',action,{seq:0,epoch:1});
    for(const options of calls){
      assert.equal(options.headers['X-Human-First-Move-At'],'1790747000.123');
      assert.equal(options.body.byteLength,5);
    }
    await upload({...run,firstMoveAt:null},[[0,0]],'browser','writer','seal',{seq:0,epoch:1});
    assert.equal(calls.at(-1).headers['X-Human-First-Move-At'],undefined);
    const timed = { ...run, wallTimeline: { version: 1, anchors: [[1,1790747000123],[2,1790747000130]], started_at_ms: 1790747000000 } };
    await upload(timed, [[0,0]], 'browser', 'writer', 'monitor', {seq:0,epoch:1});
    assert.deepEqual(JSON.parse(calls.at(-1).headers['X-Human-Wall-Timeline']).anchors, [[1,1790747000123]]);
    assert.equal(calls.at(-1).body.byteLength,5);
  } finally {globalThis.fetch=original;}
});
