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
  return {live,change(id,next=[]) {run={...run,id};events=next;},score(value){run={...run,score:value};},add(event) {events.push(event);}};
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

test('resume negotiation uploads only requested tail and flushes later moves after ready', async t=>{
  const original=globalThis.WebSocket;globalThis.WebSocket=Socket;t.after(()=>{globalThis.WebSocket=original;});
  const f=fixture(async path=>({...descriptor(path.split('/').at(-2)),resume_supported:true}));t.after(()=>f.live.dispose());
  f.change('old',[[0,100],[1,200],[2,300]]);await f.live.start();const ws=Socket.instances[0];ws.open();
  assert.equal(ws.messages.length,1);assert.equal(JSON.parse(ws.messages[0]).resume,true);
  ws.onmessage({data:JSON.stringify({type:'prefix_request',run_id:'old',start:2,seq:3})});
  f.add([3,400]);await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual([...ws.messages[1]],[72,76,80,49,0,2,44,1,0,0]);
  ws.ack('old',3);assert.deepEqual([...ws.messages[2]],[3,144,1,0,0]);
});

test('missing server state requests full history; invalid resume boundary closes socket', async t=>{
  const original=globalThis.WebSocket;globalThis.WebSocket=Socket;t.after(()=>{globalThis.WebSocket=original;});
  const f=fixture(async path=>({...descriptor(path.split('/').at(-2)),resume_supported:true}));t.after(()=>f.live.dispose());
  f.change('old',[[0,100],[1,200]]);await f.live.start();const ws=Socket.instances[0];ws.open();
  ws.onmessage({data:JSON.stringify({type:'prefix_request',run_id:'old',start:0,seq:2})});await new Promise(resolve=>setImmediate(resolve));
  assert.equal(ws.messages[1].byteLength,15);
  ws.onmessage({data:JSON.stringify({type:'prefix_request',run_id:'old',start:3,seq:2})});
  // Completed uploads ignore duplicate requests; a new handshake rejects bad boundaries.
  ws.ack('old',2);f.change('new',[[0,100]]);await f.live.runChanged();
  ws.onmessage({data:JSON.stringify({type:'prefix_request',run_id:'new',start:2,seq:1})});await new Promise(resolve=>setImmediate(resolve));assert.equal(ws.closed,true);
});

test('current score best is derived while external best changes are sent once', async t=>{
  const original=globalThis.WebSocket;globalThis.WebSocket=Socket;t.after(()=>{globalThis.WebSocket=original;});
  const f=fixture();t.after(()=>f.live.dispose());await f.live.start();const ws=Socket.instances[0];ws.open();ws.ack('old');
  f.score(1000);await new Promise(resolve=>setImmediate(resolve));const count=ws.messages.length;
  f.live.updateBest(1000);assert.equal(ws.messages.length,count);
  f.live.updateBest(2000);assert.equal(JSON.parse(ws.messages.at(-1)).best_score,2000);
  f.live.updateBest(2000);f.live.updateBest(1500);assert.equal(ws.messages.length,count+1);
});
