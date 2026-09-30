import test from 'node:test';
import assert from 'node:assert/strict';
import { createPracticeSession, PRACTICE_SESSION_KEY, PRACTICE_SESSION_TTL } from '../src/projects/practiceSession.js';

function memoryStorage() {
  const data = new Map();
  return { getItem: key => data.get(key), setItem: (key, value) => data.set(key, value) };
}
const unauthorized = () => { throw Object.assign(new Error('Guest'), {status:401}); };

test('reuses a confirmed display across reloads for five minutes, without credentials or roles', async () => {
  const storage = memoryStorage(); let time = 1000, calls = 0;
  const options = { storage, now:()=>time, readSession:async()=> {
    calls++; return {user:{id:89,display_name:'Player',site_role:'admin',token:'never-cache'}};
  }};
  const first = createPracticeSession(options);
  await first.sync(); await first.sync();
  const reloaded = createPracticeSession(options);
  assert.deepEqual(reloaded.peek(), {fresh:true,user:{id:89,display_name:'Player'}});
  await reloaded.sync(); assert.equal(calls,1);
  assert.deepEqual(JSON.parse(storage.getItem(PRACTICE_SESSION_KEY)).user,{id:89,display_name:'Player'});
  time += PRACTICE_SESSION_TTL;
  await reloaded.sync(); assert.equal(calls,2);
});

test('deduplicates concurrent checks and works without storage', async () => {
  let done, calls=0;
  const session=createPracticeSession({readSession:()=>{ calls++; return new Promise(resolve=>{done=resolve;}); }});
  const a=session.sync(), b=session.sync();
  assert.equal(a,b); done({user:{id:1,display_name:'One'}});
  await a; await session.sync(); assert.equal(calls,1);
});

test('guests do not repeatedly probe sibling sites; manual synchronization can retry', async () => {
  const storage=memoryStorage(); let bridges=0, calls=0;
  const options={storage,readSession:()=>{calls++;return unauthorized();},origins:['main','play','live'],bridge:async()=>{bridges++;}};
  await createPracticeSession(options).sync();
  assert.equal(calls,4); assert.equal(bridges,3);
  await createPracticeSession(options).sync();
  assert.equal(calls,4); assert.equal(bridges,3);
  await createPracticeSession(options).sync({force:true});
  assert.equal(bridges,6);
});

test('cookie recovery stops probing after the first successful sibling', async () => {
  let signedIn=false, bridges=0;
  const session=createPracticeSession({readSession:async()=>signedIn?{user:{id:2,display_name:'Recovered'}}:unauthorized(),
    origins:['main','play','live'],bridge:async()=>{bridges++;signedIn=true;}});
  assert.equal((await session.sync()).id,2); assert.equal(bridges,1);
  session.clear(); assert.equal(session.peek().user,null);
});

test('network errors neither start sibling probes nor cache a false logout', async () => {
  const storage=memoryStorage(); let time=1000, fail=false, bridges=0;
  const session=createPracticeSession({storage,now:()=>time,origins:['main'],bridge:async()=>{bridges++;},
    readSession:async()=>{if(fail)throw new Error('offline');return {user:{id:1,display_name:'One'}};}});
  await session.sync(); time+=PRACTICE_SESSION_TTL; fail=true;
  await assert.rejects(session.sync(),/offline/);
  assert.equal(session.peek().user.id,1); assert.equal(session.peek().fresh,false); assert.equal(bridges,0);
});
