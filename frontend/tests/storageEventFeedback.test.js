import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../src/services/storage/localStorageStore.js', import.meta.url), 'utf8').replace('export function', 'function');

test('two tabs converge after a preference change without read-triggered storage feedback', () => {
  const data = new Map(), queue = [], tabs = [];
  for (let i = 0; i < 2; i++) {
    const storage = {
      getItem: key => data.get(key) ?? null,
      setItem(key, value) {
        if (data.get(key) === value) return;
        data.set(key, value);
        queue.push(1-i);
      },
      removeItem(key) { if (data.delete(key)) queue.push(1-i); },
    };
    const context = vm.createContext({ window: { localStorage: storage } });
    vm.runInContext(source + '\nglobalThis.store = createLocalStorageStore({key:"user-preferences",defaultValue:{}});', context);
    tabs.push(context.store);
  }
  tabs[0].write({ theme: 'Verse', dark_mode: true });
  let received = 0;
  while (queue.length && received < 100) {
    const tab = tabs[queue.shift()];
    // Appearance refresh reads language and appearance from the same store.
    tab.read(); tab.read();
    received++;
  }
  assert.equal(queue.length, 0, 'a storage event must not create more storage events');
  assert.equal(received, 1);
  assert.equal(tabs[1].read().theme, 'Verse');
  for(let i=0;i<100;i++) { tabs[0].read(); tabs[1].readEnvelope(); }
  assert.equal(queue.length, 0);
  tabs[1].write({theme:'Default'});
  assert.equal(queue.length, 1, 'real edits still notify the other tab');
});

test('blocked storage getter falls back safely; full storage does not prevent reads', () => {
  const context = vm.createContext({ window: Object.defineProperty({}, 'localStorage', { get(){throw Error('blocked');} }) });
  vm.runInContext(source + '\nglobalThis.store = createLocalStorageStore({key:"settings",defaultValue:42});', context);
  assert.equal(context.store.read(), 42);
  context.store.write(43);
  context.window = {localStorage:{getItem:()=>JSON.stringify({version:1,value:44}),setItem(){throw Error('quota');}}};
  assert.equal(context.store.read(),44);
  assert.throws(()=>context.store.write(45), /quota/);
});

test('Play storage handler ignores unrelated keys and session storage but syncs real settings', () => {
  const app = readFileSync(new URL('../src/human/HumanApp.vue', import.meta.url),'utf8');
  const handler = app.slice(app.indexOf('function refreshStoredPreferences(event)'), app.indexOf('function leavePage()'));
  let appearance=0, play=0;
  const storage={};
  const context=vm.createContext({window:{localStorage:storage}, settingsStore:{key:'2048tables:human-settings'},refreshPlaySettings:()=>play++,refreshAppearance:()=>appearance++});
  vm.runInContext(handler,context);
  for(const key of ['2048tables:probe','run-lock','2048tables:account-preferences-owner'])context.refreshStoredPreferences({key,storageArea:storage});
  context.refreshStoredPreferences({key:null,storageArea:{}});
  assert.equal(appearance+play,0);
  for(const key of ['2048tables:user-preferences','saved-vth-theme-cache-v1'])context.refreshStoredPreferences({key,storageArea:storage});
  for(const key of ['2048tables:human-settings',null])context.refreshStoredPreferences({key,storageArea:storage});
  assert.equal(appearance,2);assert.equal(play,2);
  assert.match(app,/addEventListener\('storage', refreshStoredPreferences\)/);
  assert.match(app,/removeEventListener\('storage', refreshStoredPreferences\)/);
});
