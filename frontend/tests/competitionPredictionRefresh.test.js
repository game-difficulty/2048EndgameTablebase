import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {competitionPollInterval} from '../src/features/roomActivities/competitionPolling.js';
const source=readFileSync(new URL('../src/features/roomActivities/CompetitionPredictions.vue',import.meta.url),'utf8');
const body=source.slice(source.indexOf('async function refresh(){'),source.indexOf('async function open()'));
function fixture(api){
  return new Function('api',`
    let fetching=false,refreshAfter=false,refreshTimer,nextRefreshAt=0,lastRefreshAt=0,epoch=0,balanceVersion='';
    const data={value:{}},serverOffset={},kinds={value:[]},selection={};const emit=()=>{};
    const document={hidden:false};
    ${body};return {refresh,stop(){epoch++;clearTimeout(refreshTimer);}};
  `)(api);
}
const flush=async()=>{for(let i=0;i<10;i++)await Promise.resolve();};

test('prediction fallback adapts to interest, market state, and visibility',()=>{
  const base={connected:true,hidden:false,opened:false,busy:false,markets:[{status:'open'}]};
  assert.equal(competitionPollInterval(base),60000);
  assert.equal(competitionPollInterval({...base,opened:true}),5000);
  assert.equal(competitionPollInterval({...base,opened:true,markets:[{status:'closed'}]}),15000);
  assert.equal(competitionPollInterval({...base,opened:true,markets:[{status:'settled'}]}),60000);
  assert.equal(competitionPollInterval({...base,pending:{request_id:'uncertain'}}),5000);
  for(const overrides of [{hidden:true},{connected:false},{busy:true}])
    assert.equal(competitionPollInterval({...base,...overrides}),Infinity);
});

test('rapid snapshot notifications cannot create a continuous predictions request loop',async t=>{
  t.mock.timers.enable({apis:['Date','setTimeout'],now:1000});let calls=0;
  const view=fixture(async()=>{calls++;return {markets:[],server_time:1};});
  t.after(()=>view.stop());
  for(let i=0;i<100;i++)view.refresh();await flush();
  assert.equal(calls,1);
  t.mock.timers.tick(1999);await flush();assert.equal(calls,1);
  t.mock.timers.tick(1);await flush();assert.equal(calls,2);
});

test('rate limit Retry-After holds all triggers, including in-flight followups',async t=>{
  t.mock.timers.enable({apis:['Date','setTimeout'],now:1000});let calls=0;
  const view=fixture(async()=>{calls++;throw Object.assign(Error('busy'),{status:429,retryAfter:30});});
  t.after(()=>view.stop());
  view.refresh();view.refresh();await flush();
  for(let i=0;i<9;i++){t.mock.timers.tick(3000);view.refresh();await flush();}
  assert.equal(calls,1);
  t.mock.timers.tick(3000);await flush();assert.equal(calls,2);
});
