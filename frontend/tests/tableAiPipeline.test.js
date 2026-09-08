import assert from 'node:assert/strict';
import test from 'node:test';
import { TableAiCache } from '../src/features/gamer/services/tableAiCache.js';
import { createTableAiStreamTransport } from '../src/features/gamer/services/tableAiStreamTransport.js';

const tick=()=>new Promise(resolve=>setImmediate(resolve));
const body=step=>({catalog_version:'v1',full_pattern:'LL_1024',board_codes:[step,...Array(15).fill(0)],
  rng_state:[step+1,2,3,4],spawn_rate4:.1,difficulty:0,random_only:false});

// Virtual browser/cloud/Worker links exercise the real cache and transport without wall-clock sleeps.
async function simulate(playInterval,lookupMs) {
  let time=0,callbacks,routeId,opens=0,stopped=false,workerBusy=false,billing=false;
  let workerAllowed=-1,workerProduced=-1,cloudAllowed=7,cloudProduced=-1,remoteAllowed=-1;
  const events=[],buffer=new Map();
  const later=(delay,callback)=>events.push({at:time+delay,callback});
  const advance=async until=>{
    while(events.some(event=>event.at<=until)) {
      events.sort((a,b)=>a.at-b.at);
      const event=events.shift(); time=event.at; event.callback(); await tick();
    }
    time=until; await tick();
  };
  const deliver=data=>later(150,()=>{if(!stopped)callbacks.onMessage({action:'GAMER_STREAM_EVENT',data:{route_id:routeId,...data}})});
  const replenish=()=>{
    const allowed=Math.min(cloudAllowed+64,cloudProduced+64);
    if(allowed>remoteAllowed && (allowed-remoteAllowed>=4 || remoteAllowed-cloudProduced<=4)) {
      remoteAllowed=allowed;
      later(150,()=>{workerAllowed=Math.max(workerAllowed,allowed);pump()});
    }
  };
  const drain=()=>{
    if(stopped || billing || cloudProduced>=cloudAllowed || !buffer.has(cloudProduced+1))return;
    billing=true;
    later(1,()=>{
      billing=false; const seq=++cloudProduced; const item=buffer.get(seq); buffer.delete(seq);
      deliver(item); replenish(); drain();
    });
  };
  const pump=()=>{
    if(stopped || workerBusy || workerProduced>=workerAllowed)return;
    workerBusy=true;
    later(lookupMs,()=>{
      workerBusy=false; const seq=++workerProduced;
      const item={...body(seq),seq,type:'result',results:{left:.9},dtype:'float64'};
      later(150,()=>{buffer.set(seq,item);assert.ok(buffer.size<=64);drain()}); pump();
    });
  };
  const client={connect(){later(0,()=>callbacks.onOpen())},getSocket:()=>({readyState:1}),disconnect(){stopped=true},
    send(action,data) {
      if(action==='GAMER_STREAM_CANCEL'){stopped=true;return}
      later(150,()=>{
        if(action==='GAMER_STREAM_OPEN') {opens++;routeId=data.route_id;replenish()}
        if(action==='GAMER_STREAM_CREDIT') {
          assert.ok(data.allow_through-data.consumed<=64);
          cloudAllowed=Math.max(cloudAllowed,data.allow_through);
          replenish();drain();deliver({type:'window',allow_through:cloudAllowed});
        }
      });
    }};
  const transport=createTableAiStreamTransport({createClient:value=>{callbacks=value;return client},now:()=>time});
  const cache=new TableAiCache({transport,now:()=>time}),waits=[];
  try {
    for(let step=0;step<256;step++) {
      const start=time; let done=false;
      const pending=cache.lookup(body(step)).then(()=>{done=true}); await tick();
      while(!done) {
        assert.ok(events.length,'producer stalled without scheduled data');
        await advance(Math.min(...events.map(e=>e.at)));
      }
      await pending; waits.push(time-start); await advance(time+playInterval);
    }
    return {opens,first:waits[0],warmupMax:Math.max(...waits.slice(1,128)),steadyMax:Math.max(...waits.slice(128))};
  } finally {cache.close()}
}

for(const [playInterval,lookupMs] of [[20,5],[4,5],[4,15]]) {
  test(`two 300ms RTT links overlap with ${playInterval}ms playback and ${lookupMs}ms lookup`,async t=>{
    const metrics=await simulate(playInterval,lookupMs);
    t.diagnostic(JSON.stringify(metrics));
    assert.equal(metrics.opens,1);
    assert.ok(metrics.first>=600);
    assert.ok(metrics.steadyMax<=lookupMs+1, 'steady stalls must not contain another network round trip');
  });
}
