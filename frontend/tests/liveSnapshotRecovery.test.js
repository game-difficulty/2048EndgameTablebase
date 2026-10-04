import test from 'node:test';
import assert from 'node:assert/strict';
import { createSnapshotRecovery } from '../src/live/snapshotRecovery.js';
import { readFileSync } from 'node:fs';

test('the actual LivePage HTTP summary path also installs competition state',async()=>{
  const source=readFileSync(new URL('../src/live/LivePage.vue',import.meta.url),'utf8');
  const body=source.slice(source.indexOf('async function refreshSummary()'),source.indexOf('function installSnapshot(data)'));
  const seen=[];
  const refresh=new Function('api','installSnapshot',`
    let summaryRequest=0,stopped=false,chatHistoryLoaded=true;
    const room={content_kind:'competition-match'},likes={update(){}},allTime={},statsRange={value:'all'},
      chatList={value:{scrollHeight:1000,scrollTop:0,clientHeight:100}},messages={},musicUrl={value:'set'};
    const mergeLiveChat=()=>[],showNotice=()=>{},t=x=>x;
    ${body}; return refreshSummary;
  `)(async()=>({match:{phase:'GAME_B_PLAYING',content_sequence:20}}),data=>seen.push(data));
  await refresh();
  assert.equal(seen[0].type,'snapshot');assert.equal(seen[0].match.phase,'GAME_B_PLAYING');
});

test('HTTP recovery installs a real snapshot; concurrent requests coalesce',async()=>{
  let resolve,calls=0;const seen=[];
  const recovery=createSnapshotRecovery({url:'/snapshot',install:v=>seen.push(v),fetcher:()=>{calls++;return new Promise(r=>resolve=r);}});
  const first=recovery.refresh(),second=recovery.refresh();
  assert.equal(first,second);
  resolve({ok:true,json:async()=>({match:{phase:'GAME_B_PLAYING',content_sequence:20}})});
  await first;
  assert.equal(calls,1);assert.equal(seen[0].type,'snapshot');assert.equal(seen[0].match.phase,'GAME_B_PLAYING');
  recovery.stop();
});

test('a hanging HTTP body times out and a later attempt can succeed',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  let signal,calls=0;const seen=[];
  const recovery=createSnapshotRecovery({url:'/snapshot',install:v=>seen.push(v),fetcher:async(_,opts)=>{
    signal=opts.signal;return {ok:true,json:()=>++calls===1?new Promise(()=>{}):Promise.resolve({match:{content_sequence:2}})};
  }});
  const first=recovery.refresh();await Promise.resolve();
  t.mock.timers.tick(3000);await first;
  assert.equal(signal.aborted,true);
  await recovery.refresh();assert.equal(seen.length,1);
  recovery.stop();
});

test('unmount rejects a late recovery response',async()=>{
  let resolve;const seen=[];
  const recovery=createSnapshotRecovery({url:'/snapshot',install:v=>seen.push(v),fetcher:()=>new Promise(r=>resolve=r)});
  const pending=recovery.refresh();recovery.stop();
  resolve({ok:true,json:async()=>({match:{content_sequence:99}})});
  await pending;assert.deepEqual(seen,[]);
});
