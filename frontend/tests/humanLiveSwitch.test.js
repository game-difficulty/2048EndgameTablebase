import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
mock.module('../src/human/client.js', { namedExports: { json: async () => { throw Error('unexpected request'); } } });
const { createLiveBroadcast } = await import('../src/human/liveBroadcast.js');
class Socket {
  static instances = [];
  constructor() { this.readyState=0; this.messages=[]; this.closed=false; Socket.instances.push(this); }
  send(value) { this.messages.push(value); }
  close() { this.closed=true; this.readyState=3; this.onclose?.(); }
  open() { this.readyState=1; this.onopen(); }
  ack(id, seq=0, supported=true) { this.onmessage({data:JSON.stringify({type:'ready',run_id:id,seq,switch_supported:supported})}); }
}
const descriptor = id => ({room_id:'room',run_id:id,lease:id,publish_url:'ws://local'});
function fixture(request) {
  Socket.instances=[];
  let run={id:'old',variant:'4x4',epoch:1}, events=[];
  const session={liveContext:()=>({run,browser:'browser',writer:'writer'}),
    getEvents:(start=0,end=events.length)=>events.slice(start,end),getEventCount:()=>events.length,liveCheckpoint:async()=>{}};
  const live=createLiveBroadcast(session,()=>0,request || (async path=>descriptor(path.split('/').at(-2))));
  return {live,change(id,next=[]) {run={...run,id};events=next;},add(event) {events.push(event);}};
}
test('switch keeps socket and flushes new actions only after matching ready', async t=>{
  const original=globalThis.WebSocket;globalThis.WebSocket=Socket;t.after(()=>{globalThis.WebSocket=original;});
  const f=fixture();t.after(()=>f.live.dispose());
  await f.live.start();const ws=Socket.instances[0];ws.open();ws.ack('old');
  f.add([0,123]);f.live.publishTail();f.change('new');f.live.publishTail();assert.equal(ws.closed,false);
  await f.live.runChanged();assert.equal(Socket.instances.length,1);assert.equal(JSON.parse(ws.messages.at(-2)).type,'switch');
  f.add([1,456]);const count=ws.messages.length;f.live.publishTail();ws.ack('old');assert.equal(ws.messages.length,count);
  ws.ack('new');assert.equal(ws.messages.length,count+1);assert.deepEqual([...ws.messages.at(-1)],[1,200,1,0,0]);
});
test('obsolete lease response cannot switch back to an older run', async t=>{
  const original=globalThis.WebSocket;globalThis.WebSocket=Socket;t.after(()=>{globalThis.WebSocket=original;});let resolve;
  const f=fixture(async path=>{const id=path.split('/').at(-2);if(id==='middle')return new Promise(done=>{resolve=done;});return descriptor(id);});
  t.after(()=>f.live.dispose());await f.live.start();const ws=Socket.instances[0];ws.open();ws.ack('old');
  f.change('middle');const obsolete=f.live.runChanged();await new Promise(resolve=>setImmediate(resolve));f.change('new');const latest=f.live.runChanged();resolve(descriptor('middle'));await obsolete;await latest;const count=ws.messages.length;
  assert.equal(ws.messages.length,count);assert.equal(JSON.parse(ws.messages.at(-2)).lease,'new');
});
test('older server retains reconnect fallback', async t=>{
  const original=globalThis.WebSocket;globalThis.WebSocket=Socket;t.after(()=>{globalThis.WebSocket=original;});const f=fixture();t.after(()=>f.live.dispose());
  await f.live.start();const ws=Socket.instances[0];ws.open();ws.ack('old',0,false);
  f.change('new');await f.live.runChanged();assert.equal(ws.closed,true);assert.equal(Socket.instances.length,2);
});

test('next lease waits for the outstanding switch acknowledgement', async t=>{
  const original=globalThis.WebSocket;globalThis.WebSocket=Socket;t.after(()=>{globalThis.WebSocket=original;});
  const requested=[];
  const f=fixture(async path=>{const id=path.split('/').at(-2);requested.push(id);return descriptor(id);});
  t.after(()=>f.live.dispose());await f.live.start();const ws=Socket.instances[0];ws.open();ws.ack('old');
  f.change('middle');await f.live.runChanged();f.change('new');const latest=f.live.runChanged();
  await new Promise(resolve=>setImmediate(resolve));assert.deepEqual(requested,['old','middle']);
  ws.ack('middle');await latest;assert.deepEqual(requested,['old','middle','new']);
  ws.ack('new');assert.equal(Socket.instances.length,1);assert.equal(ws.closed,false);
});
