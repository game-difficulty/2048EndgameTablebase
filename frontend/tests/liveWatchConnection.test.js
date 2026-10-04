import test from 'node:test';
import assert from 'node:assert/strict';
import { createWatchConnection } from '../src/live/watchConnection.js';
import { canConnectLive, backgroundExpired } from '../src/live/pipPolicy.js';
import { createMatchDeltaDecoder } from '../src/live/matchDelta.js';

const room = { id: 'competition-demo', protocol: 'competition-match-v1', api_base: '/api/live/rooms/demo' };
const snapshot = { type: 'snapshot', room_id: room.id, protocol: room.protocol,
  match:{match_public_key:'demo',generation:1,content_sequence:1} };
const flush = async () => { for (let i=0;i<8;i++) await Promise.resolve(); };
function setup(t, overrides={}) {
  t.mock.timers.enable({apis:['Date','setTimeout','setInterval'],now:1000});
  const sockets=[],messages=[],state={connected:false,ready:false,ended:false};
  const original=globalThis.WebSocket;
  globalThis.WebSocket=class {
    constructor(){this.readyState=0;this.sent=[];sockets.push(this);}
    send(data){this.sent.push(data);}
    close(){this.readyState=2;} // Deliberately never dispatch onclose.
    open(){this.readyState=1;this.onopen();}
    message(data){return this.onmessage({data:data instanceof ArrayBuffer?data:JSON.stringify(data)});}
  };
  const connection=createWatchConnection({room,url:'wss://example.test/watch',canConnect:()=>true,expired:()=>false,
    onOpen:()=>{state.connected=true;},onDisconnect:()=>{state.connected=false;state.ready=false;},
    onSnapshot:()=>{state.ready=true;},onEnded:()=>{state.ended=true;},onMessage:data=>messages.push(data),random:()=>0,...overrides});
  t.after(()=>{connection.stop();globalThis.WebSocket=original;});
  connection.connect();
  return {connection,sockets,messages,state,tick:ms=>t.mock.timers.tick(ms)};
}

test('silent OPEN socket is detached within 30 seconds without waiting for onclose',async t=>{
  const {sockets,messages,state,tick}=setup(t);
  const old=sockets[0];old.open();await old.message(snapshot);
  assert.equal(state.ready,true);
  for(let i=0;i<30;i++)tick(1000);
  assert.equal(state.connected,false);assert.equal(state.ready,false);
  assert.equal(old.readyState,2);
  tick(1000);assert.equal(sockets.length,2);
  sockets[1].open();await sockets[1].message(snapshot);
  await old.message({type:'room_ended'});old.onclose();old.onerror();
  assert.equal(messages.length,2);assert.equal(state.ready,true);
  tick(1000);assert.equal(sockets.length,2);
});

test('handshake and first snapshot have separate 10-second deadlines',async t=>{
  const {sockets,tick}=setup(t);
  tick(10000);tick(1000);assert.equal(sockets.length,2);
  tick(9000);sockets[1].open();
  tick(9000);await sockets[1].message({type:'presence',online:true});
  tick(1000);assert.equal(sockets[1].readyState,2);
  tick(2000);assert.equal(sockets.length,3);
});

test('normal presence and binary data keep a synchronized connection alive',async t=>{
  const {sockets,state,tick}=setup(t);sockets[0].open();await sockets[0].message(snapshot);
  for(let i=0;i<36;i++){
    tick(10000);
    await sockets[0].message(i%2?new ArrayBuffer(4):{type:'presence',online:true});
  }
  assert.equal(sockets.length,1);assert.equal(state.ready,true);
  assert.equal(sockets[0].sent.length,36);
});

test('room query times out even if fetch ignores abort, then reconnects',async t=>{
  let signal;
  t.mock.method(globalThis,'fetch',(_url,options)=>{signal=options.signal;return new Promise(()=>{});});
  const {sockets,state,tick}=setup(t,{room:{...room,dynamic:true}});
  sockets[0].onclose();assert.equal(state.connected,false);
  tick(3000);await flush();assert.equal(signal.aborted,true);
  tick(1000);assert.equal(sockets.length,2);
});

test('late room query cannot end a newer connection or create another retry',async t=>{
  let respond;
  t.mock.method(globalThis,'fetch',()=>new Promise(resolve=>{respond=resolve;}));
  const {connection,sockets,state,tick}=setup(t,{room:{...room,dynamic:true}});
  sockets[0].onclose();connection.connect();
  sockets[1].open();await sockets[1].message(snapshot);
  respond({status:404});await flush();tick(2000);
  assert.equal(state.ended,false);assert.equal(state.ready,true);assert.equal(sockets.length,2);
});

test('only a current explicit 404 ends a dynamic room',async t=>{
  t.mock.method(globalThis,'fetch',async()=>({status:404}));
  const {sockets,state,tick}=setup(t,{room:{...room,dynamic:true}});
  sockets[0].onclose();await flush();tick(60000);
  assert.equal(state.ended,true);assert.equal(sockets.length,1);
});

test('failed room lookup still retries and unmount cancels pending probes',async t=>{
  t.mock.method(globalThis,'fetch',async()=>{throw Error('offline');});
  const {connection,sockets,state,tick}=setup(t,{room:{...room,dynamic:true}});
  sockets[0].onclose();await flush();tick(1000);assert.equal(sockets.length,2);
  sockets[1].onclose();connection.stop();await flush();tick(60000);
  assert.equal(sockets.length,2);assert.equal(state.ended,false);
});

test('malformed or wrong-room snapshots never mark a connection synchronized',async t=>{
  const {sockets,state,tick}=setup(t);sockets[0].open();
  await sockets[0].message({...snapshot,room_id:'another-room'});
  assert.equal(state.ready,false);tick(1000);sockets[1].open();
  await sockets[1].onmessage({data:'not-json'});
  assert.equal(state.connected,false);
});

test('snapshot installation failure triggers recovery instead of resetting backoff',async t=>{
  const {sockets,state,tick}=setup(t,{onMessage:()=>{throw Error('broken renderer');}});
  sockets[0].open();await sockets[0].message(snapshot);tick(1000);
  sockets[1].open();await sockets[1].message(snapshot);tick(1000);
  assert.equal(sockets.length,2);assert.equal(state.ready,false);
  tick(1000);assert.equal(sockets.length,3);
});

test('successful snapshot resets retry backoff, not socket open',async t=>{
  const {sockets,tick}=setup(t);
  sockets[0].open();tick(10000);tick(1000);
  sockets[1].open();tick(10000);tick(1000);assert.equal(sockets.length,2);
  tick(1000);assert.equal(sockets.length,3);
  sockets[2].open();await sockets[2].message(snapshot);sockets[2].onclose();
  tick(1000);assert.equal(sockets.length,4);
});

test('background policy pauses retries, PiP permits recovery, and foreground checks stale sockets',async t=>{
  let hidden=false,pip=false,deadline=0;
  const {connection,sockets,tick}=setup(t,{canConnect:()=>canConnectLive(hidden,pip),expired:()=>backgroundExpired(hidden,pip,deadline,Date.now())});
  sockets[0].open();await sockets[0].message(snapshot);
  hidden=true;sockets[0].onclose();tick(60000);assert.equal(sockets.length,1);
  pip=true;connection.connect();assert.equal(sockets.length,2);
  sockets[1].open();await sockets[1].message(snapshot);
  pip=false;deadline=Date.now()+1000;tick(1000);assert.equal(sockets[1].readyState,2);
  hidden=false;connection.connect();assert.equal(sockets.length,3);
  sockets[2].open();await sockets[2].message(snapshot);
  // A foreground check must not wait for the next throttled browser timer.
  t.mock.timers.setTime(Date.now()+31000);connection.check();
  assert.equal(sockets[2].readyState,2);
});

test('presence cannot suppress periodic match resync and a newer watermark triggers recovery',async t=>{
  let requests=0;
  const {connection,sockets,tick}=setup(t,{onResync:()=>{requests++;}});
  sockets[0].open();await sockets[0].message(snapshot);await flush();
  assert.equal(requests,1);
  for(let i=0;i<12;i++){tick(5000);await sockets[0].message({type:'presence',online:true});await flush();}
  assert.ok(requests>=5);
  tick(3000);
  await sockets[0].message({type:'match_watermark',match_public_key:'demo',generation:1,content_sequence:2});await flush();
  const count=requests;
  connection.observeSnapshot({...snapshot,match:{...snapshot.match,content_sequence:2}});
  tick(3000);
  await sockets[0].message({type:'match_watermark',match_public_key:'demo',generation:1,content_sequence:2});await flush();
  assert.equal(requests,count);
  assert.equal(sockets.length,1);
});

test('empty competition snapshots do not claim to have synchronized',async t=>{
  const {sockets,state,tick}=setup(t);
  sockets[0].open();await sockets[0].message({...snapshot,match:null});
  assert.equal(state.ready,false);
  tick(10000);assert.equal(sockets[0].readyState,2);
});

test('social channel synchronizes without a board and never starts board HTTP recovery',async t=>{
  let recoveries=0;
  const {sockets,state,tick}=setup(t,{channel:'social',onResync:()=>{recoveries++;}});
  sockets[0].open();await sockets[0].message({type:'social_snapshot',room_id:room.id,protocol:room.protocol,online:true});
  assert.equal(state.ready,true);
  for(let i=0;i<12;i++){tick(10000);await sockets[0].message({type:'pong'});await flush();}
  assert.equal(recoveries,0);assert.equal(sockets.length,1);assert.equal(sockets[0].sent.length,12);
});

test('a board delta gap reconnects only the board; social gifts continue immediately',async t=>{
  const {sockets,tick}=setup(t,{channel:'board',decoder:createMatchDeltaDecoder(room)});
  const received=[];
  const social=createWatchConnection({room,url:'wss://example.test/watch?channel=social',channel:'social',
    canConnect:()=>true,expired:()=>false,onMessage:data=>received.push(data),random:()=>0});
  t.after(()=>social.stop());social.connect();
  const board=sockets[0],interactive=sockets[1];board.open();interactive.open();
  await board.message({...snapshot,stream_epoch:'one',stream_sequence:1});
  await interactive.message({type:'social_snapshot',room_id:room.id,protocol:room.protocol});
  await board.message({type:'match_delta',room_id:room.id,protocol:room.protocol,stream_epoch:'one',base_sequence:2,stream_sequence:3});
  assert.equal(board.readyState,2);assert.equal(interactive.readyState,1);
  await interactive.message({type:'gift',id:'paid'});assert.equal(received.at(-1).id,'paid');
  tick(1000);assert.equal(sockets.length,3);
  const restarted=sockets[2];restarted.open();await restarted.message({...snapshot,stream_epoch:'two',stream_sequence:4});
  assert.equal(restarted.readyState,1);assert.equal(interactive.readyState,1);
});
