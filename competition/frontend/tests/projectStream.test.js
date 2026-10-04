import test from 'node:test';
import assert from 'node:assert/strict';
import { ProjectStreamSender, PROJECT_STREAM_PROTOCOL } from '../src/projects/projectStream.js';
import { checkpointDelta } from '../src/projects/checkpointDelta.js';

const packet = n => ({ instance_id: 'A:yellow', sequence: n, elapsed_ms: n * 30, finished: false,
  payload: { board: [[2]], move_count: n },
  checkpoint: { version: 1, state: { randomState: n, undo: Array.from({ length: n }, (_,i) => [i, 2, 4]) }, metric_history: [[0,0]] } });
function fixture() {
  const sockets=[],acks=[],errors=[];
  const sender=new ProjectStreamSender({
    interval:10000, createSocket:()=>{
      const socket={ sent:[],readyState:1,bufferedAmount:0,
        send(text){this.sent.push(JSON.parse(text));}, close(code,reason){this.onclose?.({code,reason});} };
      sockets.push(socket);return socket;
    },authenticate:()=>({protocol:PROJECT_STREAM_PROTOCOL}),phaseToken:()=> 'phase',
    onAck:ack=>acks.push(ack.accepted_sequence),onError:error=>errors.push(error),
  });
  const receive=message=>sockets.at(-1).onmessage({data:JSON.stringify(message)});
  receive({type:'stream.ready',protocol:PROJECT_STREAM_PROTOCOL,accepted_sequence:0});
  return {sender,sockets,receive,acks,errors};
}
test('RTT does not gate each batch; in-flight window is bounded and ACK is cumulative',()=>{
  const {sender,sockets,receive}=fixture();
  try{
    for(let i=1;i<=5;i++){sender.push(packet(i));sender.flush();}
    assert.equal(sockets[0].sent.length,4);
    assert.equal(sender.frames.length,5);
    receive({type:'stream.ack',accepted_sequence:3});sender.flush();
    assert.equal(sockets[0].sent.length,5);
    assert.deepEqual(sender.frames.map(f=>f.sequence),[4,5]);
    assert.equal(sockets[0].sent[1].data.checkpoint.delta_base,1);
  }finally{sender.close();}
});
test('reconnect sends a full base and every still-unacknowledged public frame',()=>{
  const {sender,sockets,receive}=fixture();
  try{
    for(let i=1;i<=8;i++){sender.push(packet(i));sender.flush();}
    clearInterval(sender.heartbeat);sender.open();
    receive({type:'stream.ready',protocol:PROJECT_STREAM_PROTOCOL,accepted_sequence:2});sender.flush();
    const data=sockets[1].sent[0].data;
    assert.deepEqual(data.frames.map(f=>f.sequence),[3,4,5,6,7,8]);
    assert.equal(data.checkpoint.delta_base,undefined);
  }finally{sender.close();}
});
test('a long undo history is sent as a tiny suffix between full recovery checkpoints',()=>{
  const before=packet(2000),after=packet(2001);
  const delta=checkpointDelta(before.checkpoint,after.checkpoint,2000);
  assert.equal(delta.lists.undo.keep,2000);
  assert.deepEqual(delta.lists.undo.append,[[2000,2,4]]);
  assert.ok(JSON.stringify(delta).length<JSON.stringify(after.checkpoint).length/20);
  const undo=packet(1998);
  assert.deepEqual(checkpointDelta(before.checkpoint,undo.checkpoint,2000).lists.undo,{keep:1998,append:[]});
});

test('a stopped match closes cleanly even when the pending sequence was not accepted',()=>{
  const {sender,receive}=fixture();
  sender.push(packet(1));sender.flush();
  receive({type:'stream.ack',accepted_sequence:0,stopped:true,resync_room:true});
  assert.equal(sender.closed,true);
  assert.equal(sender.inflight.size,0);
  assert.equal(sender.reconnect,undefined);
});

test('withheld ACKs never block new local states; reconnect retains the final state and all 50 frames',()=>{
  const {sender,sockets,receive}=fixture();
  try{
    for(let i=1;i<=50;i++){sender.push({...packet(i),finished:i===50});sender.flush();}
    assert.equal(sender.latest.sequence,50);
    assert.equal(sockets[0].sent.length,4);
    clearInterval(sender.heartbeat);sender.open();
    receive({type:'stream.ready',protocol:PROJECT_STREAM_PROTOCOL,accepted_sequence:0});sender.flush();
    const data=sockets[1].sent[0].data;
    assert.equal(data.sequence,50);
    assert.equal(data.finished,true);
    assert.deepEqual(data.frames.map(f=>f.sequence),Array.from({length:50},(_,i)=>i+1));
    receive({type:'stream.ack',accepted_sequence:50});
    assert.equal(sender.frames.length,0);
  }finally{sender.close();}
});

test('a blackholed connection retries without waiting for the browser close handshake',t=>{
  t.mock.timers.enable({apis:['Date','setTimeout','setInterval'],now:1000});
  const {sender,sockets,receive}=fixture();
  try {
    sender.push(packet(1));sender.flush();
    const old=sockets[0];old.close=()=>{old.readyState=2;};
    t.mock.timers.tick(10000);
    assert.equal(sender.ready,false);
    t.mock.timers.tick(500);
    assert.equal(sockets.length,2);
    receive({type:'stream.ready',protocol:PROJECT_STREAM_PROTOCOL,accepted_sequence:0});
    sender.flush();
    assert.equal(sockets[1].sent[0].data.sequence,1);
    old.onclose({code:4409});
    assert.equal(sender.closed,false,'a delayed close from the old socket cannot block the new sender');
  } finally { sender.close(); }
});

test('a socket stuck before its first ready response automatically retries',t=>{
  t.mock.timers.enable({apis:['Date','setTimeout','setInterval'],now:1000});
  const {sender,sockets}=fixture();
  try {
    sender.open(); // Connected transport, but no stream.ready this time.
    sockets[1].close=()=>{};
    t.mock.timers.tick(12000);t.mock.timers.tick(500);
    assert.equal(sockets.length,3);
  } finally { sender.close(); }
});

test('temporary active-player denial resyncs and retries, another-page takeover remains blocked',t=>{
  t.mock.timers.enable({apis:['Date','setTimeout','setInterval'],now:1000});
  const {sender,sockets,errors}=fixture();
  try {
    sockets[0].onclose({code:4403,reason:'active_player_required'});
    assert.equal(sender.closed,false);
    assert.equal(errors.at(-1).code,'STREAM_DISCONNECTED');
    t.mock.timers.tick(500);
    sockets[1].onclose({code:4409,reason:'publisher_replaced'});
    assert.equal(sender.closed,true);
    sender.recover();t.mock.timers.tick(20000);
    assert.equal(sockets.length,2);
  } finally { sender.close(); }
});

test('healthy ordered transport keeps delta uploads even after the former two-second checkpoint interval',t=>{
  t.mock.timers.enable({apis:['Date','setTimeout','setInterval'],now:1000});
  const {sender,sockets,receive}=fixture();
  try{
    sender.push(packet(2000));sender.flush();
    receive({type:'stream.ack',accepted_sequence:2000});
    t.mock.timers.tick(3000);
    sender.push(packet(2001));sender.flush();
    const batches=sockets[0].sent.filter(m=>m.type==='project.batch');
    assert.equal(batches[1].data.checkpoint.delta_base,2000);
    assert.ok(JSON.stringify(batches[1]).length<JSON.stringify(batches[0]).length/5);
  }finally{sender.close();}
});
