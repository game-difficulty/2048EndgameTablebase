import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// Load the actual browser transport with an empty Vite environment.
const source = readFileSync(new URL('../src/api.js', import.meta.url), 'utf8').replaceAll('import.meta.env', '({})');
const { connectRoom } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('room recovery escapes CLOSING and ignores late messages from the detached socket', t => {
  t.mock.timers.enable({ apis: ['Date', 'setTimeout', 'setInterval'], now: 1000 });
  const sockets=[], messages=[];
  class Socket {
    constructor() { this.readyState=0; this.sent=[]; sockets.push(this); }
    send(text) { this.sent.push(JSON.parse(text)); }
    close() { this.readyState=2; } // No close event: simulate a dead connection.
  }
  const previousWindow=globalThis.window, previousSocket=globalThis.WebSocket;
  globalThis.window={ location:{href:'https://example.test/rooms/ABC'},localStorage:{getItem:()=>null},sessionStorage:{getItem:()=>null},setTimeout,clearTimeout };
  globalThis.WebSocket=Socket;
  const close=connectRoom('ABC',{onMessage:m=>messages.push(m)});
  try {
    const first=sockets[0];first.readyState=1;first.onopen();
    first.onmessage({data:JSON.stringify({type:'room.snapshot',data:{version:1}})});
    t.mock.timers.tick(32000);t.mock.timers.tick(1200);
    assert.equal(sockets.length,2);
    first.onmessage({data:JSON.stringify({type:'room.snapshot',data:{version:0}})});
    first.onclose({code:4403});
    assert.equal(messages.length,1);
    const next=sockets[1];next.readyState=1;next.onopen();
    next.onmessage({data:JSON.stringify({type:'room.snapshot',data:{version:2}})});
    close.resync();
    assert.equal(next.sent.at(-1).type,'room.resync');
    assert.equal(messages.at(-1).data.version,2);
  } finally {
    close();globalThis.window=previousWindow;globalThis.WebSocket=previousSocket;
  }
});

test('room authentication without an initial snapshot retries even if pong messages arrive', t => {
  t.mock.timers.enable({ apis: ['Date', 'setTimeout', 'setInterval'], now: 1000 });
  const sockets=[];
  class Socket {
    constructor() { this.readyState=0;sockets.push(this); }
    send() {}
    close() { this.readyState=2; }
  }
  const previousWindow=globalThis.window, previousSocket=globalThis.WebSocket;
  globalThis.window={ location:{href:'https://example.test/rooms/ABC'},localStorage:{getItem:()=>null},sessionStorage:{getItem:()=>null},setTimeout,clearTimeout };
  globalThis.WebSocket=Socket;
  const close=connectRoom('ABC');
  try {
    sockets[0].readyState=1;sockets[0].onopen();
    sockets[0].onmessage({data:JSON.stringify({type:'pong'})});
    t.mock.timers.tick(12000);t.mock.timers.tick(1200);
    assert.equal(sockets.length,2);
  } finally { close();globalThis.window=previousWindow;globalThis.WebSocket=previousSocket; }
});

test('a healthy room connection periodically refreshes control state, not just its heartbeat', t => {
  t.mock.timers.enable({ apis: ['Date', 'setTimeout', 'setInterval'], now: 1000 });
  const messages=[],sent=[],sockets=[];
  class Socket {
    constructor() { this.readyState=1;sockets.push(this); }
    send(text) {
      const message=JSON.parse(text);sent.push(message.type);
      if(message.type==='room.resync')this.onmessage({data:JSON.stringify({type:'room.snapshot',data:{version:2}})});
      else if(message.type==='ping')this.onmessage({data:JSON.stringify({type:'pong'})});
    }
    close() { this.readyState=3; }
  }
  const previousWindow=globalThis.window, previousSocket=globalThis.WebSocket;
  globalThis.window={ location:{href:'https://example.test/rooms/ABC'},localStorage:{getItem:()=>null},sessionStorage:{getItem:()=>null},setTimeout,clearTimeout };
  globalThis.WebSocket=Socket;
  const close=connectRoom('ABC',{onMessage:m=>messages.push(m)});
  try {
    sockets[0].onopen();
    sockets[0].onmessage({data:JSON.stringify({type:'room.snapshot',data:{version:1}})});
    t.mock.timers.tick(16000);
    assert.ok(sent.includes('room.resync'));
    assert.equal(messages.filter(m=>m.type==='room.snapshot').at(-1).data.version,2);
    assert.equal(sockets.length,1,'resync uses the healthy existing connection');
  } finally { close();globalThis.window=previousWindow;globalThis.WebSocket=previousSocket; }
});

test('AI worker timeout rejects its pending operation and a later input starts a fresh worker', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const previousWorker=globalThis.Worker, workers=[];
  class Worker {
    constructor() { this.handlers={};workers.push(this); }
    addEventListener(type,fn) { this.handlers[type]=fn; }
    postMessage(data) { this.data=data; }
    terminate() { this.terminated=true; }
  }
  globalThis.Worker=Worker;
  try {
    const { evilSpawn } = await import('../src/projects/evilSpawn.js');
    const pending=evilSpawn(Array(16).fill(0),4,123);
    const rejection=assert.rejects(pending,/暂时未响应/);
    t.mock.timers.tick(20000);await rejection;
    assert.equal(workers[0].terminated,true);
    const retry=evilSpawn(Array(16).fill(0),4,123);
    workers[1].handlers.message({data:{id:workers[1].data.id,result:{index:0,value:2}}});
    assert.deepEqual(await retry,{index:0,value:2});
  } finally { globalThis.Worker=previousWorker; }
});
