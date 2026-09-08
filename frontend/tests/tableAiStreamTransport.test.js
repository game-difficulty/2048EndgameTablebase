import assert from 'node:assert/strict';
import test from 'node:test';
import { createTableAiStreamTransport } from '../src/features/gamer/services/tableAiStreamTransport.js';

function setup() {
  const sent=[], results=[], balances=[], errors=[], latencies=[];
  let callbacks, connected=false, now=0;
  const client={send:(action,data)=>sent.push({action,data}),connect(){},disconnect(){connected=false;},
    getSocket:()=>({readyState:connected?1:0})};
  const transport=createTableAiStreamTransport({createClient:(value)=>{callbacks=value;return client;},
    onBalance:value=>balances.push(value),now:()=>now});
  const open=(id='route-1')=>transport.open({request_id:id},
    {onResult:value=>results.push(value),onError:error=>errors.push(error),onEnd:()=>results.push('end'),onLatency:ms=>latencies.push(ms)});
  const connect=()=>{connected=true;callbacks.onOpen();};
  const receive=(data)=>callbacks.onMessage({action:'GAMER_STREAM_EVENT',data});
  return {transport,sent,results,balances,errors,latencies,open,connect,receive,disconnect:()=>{connected=false;},advance:(ms=100)=>{now+=ms;}};
}

test('latency measures newly authorized results, not cloud window ACKs or replay', () => {
  const f=setup(); const handle=f.open(); f.connect();
  const receive=(seq)=>f.receive({route_id:'route-1',type:'result',seq});
  receive(0); handle.credit(0,30);
  f.advance(300); f.receive({route_id:'route-1',type:'window',allow_through:30});
  for(let seq=1;seq<=7;seq++)receive(seq);
  assert.deepEqual(f.latencies,[]);
  f.advance(310); receive(8); receive(8);
  assert.deepEqual(f.latencies,[610]);
  handle.credit(8,40); f.disconnect(); f.connect(); f.advance(); receive(9);
  assert.deepEqual(f.latencies,[610], 'replayed paid frames cannot measure replenishment');
  f.transport.close();
});

test('open waits for authenticated socket ordering; reconnect uses the same ID and receipt', () => {
  const f=setup(); const handle=f.open();
  assert.equal(f.sent.length,0);
  f.connect(); assert.equal(f.sent[0].action,'GAMER_STREAM_OPEN');
  assert.equal(f.sent[0].data.resume,false);
  f.receive({route_id:'route-1',type:'result',seq:0,token_balance:{total:90}});
  handle.credit(0,16);
  f.disconnect(); handle.credit(0,20); f.connect();
  const opens=f.sent.filter(message=>message.action==='GAMER_STREAM_OPEN');
  assert.equal(opens.length,2);
  assert.equal(opens[1].data.received,0); assert.equal(opens[1].data.resume,true);
  assert.equal(opens[1].data.route_id,'route-1');
  assert.equal(f.sent.at(-1).data.allow_through,20);
  f.transport.close();
});

test('replayed frames and late messages from cancelled routes do not update results or balances', () => {
  const f=setup(); const handle=f.open(); f.connect();
  const result={route_id:'route-1',type:'result',seq:0,token_balance:{total:90}};
  f.receive(result); f.receive(result);
  assert.equal(f.results.length,1); assert.equal(f.balances.length,1);
  handle.cancel(); f.open('route-2'); f.receive({...result,seq:1});
  assert.equal(f.results.length,1);
  f.receive({route_id:'route-2',type:'result',seq:0});
  assert.equal(f.results.length,2);
  f.transport.close();
});

test('lost server subscription ends cleanly; quota and sequence errors reach the consumer', () => {
  const f=setup(); f.open(); f.connect();
  f.receive({route_id:'route-1',type:'error',status:409,detail:'STREAM_GONE'});
  assert.deepEqual(f.results,['end']);
  f.open('route-2'); f.receive({route_id:'route-2',type:'error',status:402,detail:{code:'INSUFFICIENT_TOKENS'}});
  assert.equal(f.errors[0].status,402);
  f.open('route-3'); f.receive({route_id:'route-3',type:'result',seq:2});
  assert.equal(f.errors.length,2);
  f.transport.close();
});
