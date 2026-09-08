import assert from 'node:assert/strict';
import test from 'node:test';
import { TableAiCache } from '../src/features/gamer/services/tableAiCache.js';
import { TableDispatcher } from '../src/features/gamer/engine/tableDispatcher.js';
import { createOrdinaryRng, planGamerSpawn, copySpawnRng } from '../src/features/gamer/engine/gamerSpawn.js';

const body = (step = 0) => ({ catalog_version: 'v1', full_pattern: 'L3_256',
  board_codes: [step, ...Array(15).fill(0)], rng_state: [step + 1, 2, 3, 4],
  spawn_rate4: .1, difficulty: 0, random_only: false });
const result = (step) => ({ ...body(step), seq: step, type: 'result', results: { left: .9 }, dtype: 'float64' });
const tick = () => new Promise((resolve) => setImmediate(resolve));
function controlledTransport() {
  const requests = [];
  const transport = { open(request, callbacks) {
    const task = { request, ...callbacks, credits: [], cancelled: false };
    requests.push(task);
    return { credit: (consumed, allowed) => task.credits.push({consumed, allowed}),
      cancel: () => { task.cancelled = true; } };
  }, close() {} };
  return { requests, transport };
}

test('one subscription serves sixty-four moves without a four-step boundary', async () => {
  let now = 0;
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport, now: () => now });
  for (let step = 0; step < 64; step++) {
    const pending = cache.lookup(body(step));
    assert.equal(requests.length, 1);
    now += 50;
    requests[0].onResult(result(step));
    assert.deepEqual(await pending, result(step));
  }
  assert.ok(requests[0].credits.length > 0);
  assert.ok(requests[0].credits.length < 32, 'credits are cumulative, not one per step');
  assert.ok(requests[0].credits.every(({consumed,allowed}) => allowed-consumed <= 64));
  assert.equal(requests[0].cancelled, false);
  cache.clear();
});

test('a 300ms RTT is overlapped with playback after the initial window adapts', async () => {
  let time=0, allowed=7, produced=-1, pumping=false, opens=0, stopped=false;
  const events=[];
  const later=(delay,callback)=>events.push({at:time+delay,callback});
  const advance=async (until)=>{
    while(events.some(event=>event.at<=until)) {
      events.sort((a,b)=>a.at-b.at);
      const event=events.shift(); time=event.at; event.callback(); await tick();
    }
    time=until; await tick();
  };
  const transport={open(_body,callbacks){
    opens++;
    const pump=()=>{
      if(pumping || stopped || produced>=allowed)return;
      pumping=true;
      later(5,()=>{
        pumping=false;
        const seq=++produced;
        later(150,()=>{if(!stopped)callbacks.onResult(result(seq))});
        pump();
      });
    };
    later(150,pump);
    return {credit(_consumed,limit){later(150,()=>{
      allowed=limit; pump(); later(150,()=>callbacks.onLatency(300));
    })},cancel(){stopped=true}};
  },close(){}};
  const cache=new TableAiCache({transport,now:()=>time});
  const waits=[];
  try {
    for(let step=0;step<80;step++) {
      const start=time;
      let resolved=false;
      const pending=cache.lookup(body(step)).then(value=>{resolved=true;return value});
      await tick();
      while(!resolved) {
        assert.ok(events.length,'route stalled');
        await advance(Math.min(...events.map(event=>event.at)));
      }
      await pending; waits.push(time-start);
      await advance(time+20);
    }
    assert.equal(opens,1);
    assert.ok(waits[0]>=300);
    assert.ok(Math.max(...waits.slice(20))<=5,`steady waits must not exceed one native lookup: ${waits}`);
  } finally {cache.close()}
});

test('cached moves proceed while the stream remains open and credit acknowledgements are missing', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const first = cache.lookup(body());
  for (let step=0; step<8; step++) requests[0].onResult(result(step));
  await first;
  for (let step=1; step<8; step++) assert.deepEqual(await cache.lookup(body(step)), result(step));
  assert.equal(requests.length,1);
  assert.ok(requests[0].credits.length > 0);
  cache.clear();
});

test('pausing cancels computation, preserves paid results and resumes past the cached anchor', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const first = cache.lookup(body()); requests[0].onResult(result(0)); await first;
  cache.cancelPrefetch();
  assert.equal(requests[0].cancelled,true);
  assert.deepEqual(await cache.lookup(body()),result(0));
  assert.equal(requests[1].request.advance_first,true);
  cache.clear();
});

test('late frames advance credit after playback has already consumed a long cached prefix', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  for (let step=0; step<=20; step++) cache.entries.set(cache.key(body(step)), {time:Date.now(), value:result(step)});
  try {
    for (let step=0; step<=20; step++) await cache.lookup(body(step));
    assert.equal(requests[0].request.advance_first,true);
    const pending=cache.lookup(body(21));
    for (let step=1; step<=8; step++) requests[0].onResult({...result(step),seq:step-1});
    assert.ok(requests[0].credits.at(-1).allowed>=20, 'late consumed nodes must release the initial window');
    for (let step=9; step<=21; step++) requests[0].onResult({...result(step),seq:step-1});
    assert.deepEqual((await pending).board_codes,body(21).board_codes);
    assert.equal(requests.length,1);
  } finally {cache.close()}
});

test('network waiting does not inflate the consumer interval', async () => {
  let now=0;
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport,now:()=>now });
  const first=cache.lookup(body());
  now=600; requests[0].onResult(result(0)); await first;
  now=604; const second=cache.lookup(body(1));
  now=1204; requests[0].onResult(result(1)); await second;
  assert.equal(cache.active.interval,4);
  assert.equal(cache.active.lastReturn,1204);
  cache.close();
});

test('switching patterns cancels the old subscription without awaiting its completion', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const first = cache.lookup(body()); requests[0].onResult(result(0)); await first;
  const next = cache.lookup({...body(1),full_pattern:'LL_1024'});
  assert.equal(requests[0].cancelled,true);
  requests[1].onResult({...result(1),seq:0,full_pattern:'LL_1024'});
  await next;
  cache.clear();
});

test('clear rejects outstanding consumers and ignores late responses', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const pending = cache.lookup(body()); cache.clear();
  requests[0].onResult(result(0));
  await assert.rejects(pending);
  assert.equal(cache.entries.size,0);
});

test('negative results and route end never cause an endless refill loop', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const pending = cache.lookup(body());
  requests[0].onResult({...result(0),results:{left:null}}); requests[0].onEnd();
  await pending; await tick();
  assert.equal(requests.length,1);
  assert.equal(cache.active.done,true);
  cache.clear();
});

test('stream failures release a waiting lookup instead of leaving AI busy', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const pending = cache.lookup(body()); requests[0].onError(Object.assign(new Error('quota'),{status:402}));
  await assert.rejects(pending,{status:402});
  cache.close();
});

test('cached rates can be reused with a different RNG but do not credit the old branch', async () => {
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({ transport });
  const pending = cache.lookup(body()); requests[0].onResult(result(0)); await pending;
  const different = {...body(),rng_state:[42,2,3,4]};
  assert.deepEqual(await cache.lookup(different),result(0));
  assert.equal(requests[0].credits.length,0);
  cache.clear();
});

test('LRU stays in memory, is bounded to 256 entries and expires at five minutes', async () => {
  let now = 0;
  const { requests, transport } = controlledTransport();
  const cache = new TableAiCache({transport,now:()=>now});
  const pending=cache.lookup(body());
  for(let index=0;index<260;index++) requests[0].onResult(result(index));
  cache.cancelPrefetch(); await assert.rejects(pending);
  assert.equal(cache.entries.size,256); assert.equal(cache.get(body()),null);
  assert.ok(cache.get(body(259)));
  now=300001; assert.equal(cache.get(body(259)),null);
});

test('predicting a spawn never advances live RNG and does not predict EvilGen', () => {
  const rng=createOrdinaryRng(), state=rng.exportState(), clone=copySpawnRng(rng);
  const board=[2,0,4,0,...Array(12).fill(0)];
  assert.deepEqual(planGamerSpawn(board,rng),planGamerSpawn(board,clone));
  assert.deepEqual(rng.exportState(),state);
  assert.equal(planGamerSpawn(board,rng,{difficulty:100}).evil,true);
});

test('variant and wrong spawn rate stay excluded; endgame policy and cooldown are unchanged', () => {
  const dispatcher=new TableDispatcher([
    {pattern:'3x3',spawnRate:.1,ai:{compatible:false}},
    {pattern:'L3',spawnRate:.2,ai:{compatible:true,policy_version:1}},
  ],.1);
  assert.equal(dispatcher.tables.length,0);
  dispatcher.reset([128,128,...Array(14).fill(0)]);
  assert.equal(dispatcher.accept({table:{target:'256',fullPattern:'L3_256'},type:1},
    {results:{left:1},dtype:'uint32'}),'AI');
  for(let index=0;index<20;index++) dispatcher.reset(dispatcher.board);
  assert.equal(dispatcher.cooldowns.size,0);
});
