import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { TAB_IDS, TAB_REGISTRY } from '../src/app/tabRegistry.js';
import { currentSite, siteProfile, tabUrl, isTableTab, MAIN_HOME_ENTRIES, TABLES_HOME_ENTRIES } from '../src/app/siteProfile.js';
import { trustedSiteOrigin, validHandoff, openSiteTab, receiveSiteContext } from '../src/app/siteNavigation.js';
import { useTabManager } from '../src/app/useTabManager.js';

test('two home profiles have six distinct entries; all table tools share one registry', () => {
  assert.equal(currentSite, 'main');
  assert.equal(MAIN_HOME_ENTRIES.length, 6);
  assert.equal(TABLES_HOME_ENTRIES.length, 6);
  assert.deepEqual(TABLES_HOME_ENTRIES.map(e => e.tab), [TAB_IDS.TRAINER,TAB_IDS.TESTER,TAB_IDS.BATTLE,TAB_IDS.REPLAY,TAB_IDS.ANALYSIS,TAB_IDS.SETTINGS]);
  assert.equal(MAIN_HOME_ENTRIES.filter(e => e.site === 'tables').length, 1);
  for (const entry of TABLES_HOME_ENTRIES) assert.ok(TAB_REGISTRY[entry.tab]);
  assert.equal(isTableTab(TAB_IDS.GAMER), false);
});
test('site detection and new deep links work in production and local preview', () => {
  assert.equal(siteProfile(new URL('https://tables.2048tables.online/')), 'tables');
  assert.equal(siteProfile(new URL('http://localhost:5188/tables/')), 'tables');
  assert.equal(siteProfile(new URL('https://2048tables.online/')), 'main');
  assert.equal(tabUrl(TAB_IDS.REPLAY, 'tables', new URL('https://2048tables.online/')).href, 'https://tables.2048tables.online/?tab=replay');
  assert.equal(tabUrl(TAB_IDS.TRAINER, 'tables', new URL('http://localhost:5188/?backend_port=8000')).href, 'http://localhost:5188/tables/?backend_port=8000&tab=trainer');
});
test('analysis tab is unique, preserves other tabs, and closing last work tab returns home', () => {
  const tabs = useTabManager();
  tabs.openTab(TAB_IDS.ANALYSIS); tabs.openTab(TAB_IDS.SETTINGS); tabs.openTab(TAB_IDS.ANALYSIS);
  assert.equal(tabs.openTabs.value.filter(t => t === TAB_IDS.ANALYSIS).length, 1);
  tabs.closeTab(TAB_IDS.ANALYSIS); tabs.closeTab(TAB_IDS.SETTINGS);
  assert.equal(tabs.activeTab.value, TAB_IDS.MAIN_MENU);
});
test('cross-window handoff requires exact origin, source, token and message type', () => {
  const loc = new URL('https://tables.2048tables.online/'), source = {};
  assert.equal(trustedSiteOrigin('https://2048tables.online.evil.test', loc), false);
  assert.equal(trustedSiteOrigin('https://2048tables.online', loc), true);
  assert.equal(trustedSiteOrigin('http://localhost:5188', loc), false);
  const event = { source, origin: loc.origin, data: { token: 'abc', type: 'tables-context' } };
  assert.equal(validHandoff(event, source, loc.origin, 'abc', 'tables-context'), true);
  for (const change of [{source:{}}, {origin:'https://evil.test'}, {data:{token:'bad',type:'tables-context'}}]) {
    assert.equal(validHandoff({...event,...change},source,loc.origin,'abc','tables-context'), false);
  }
});
function windowMock(url) {
  const listeners = new Map(), timers = new Map(); let count = 0;
  return { location:new URL(url), crypto:{randomUUID:()=> 'unique-token'}, listeners, timers,
    addEventListener:(key,fn)=>listeners.set(key,fn), removeEventListener:(key)=>listeners.delete(key),
    setTimeout:fn=>{timers.set(++count,fn);return count;}, clearTimeout:id=>timers.delete(id),
    history:{replaceState(_s,_t,url){this.url=url.href;}},
  };
}
test('handoff delivers the full training context and cleans up after acknowledgment', () => {
  const win = windowMock('https://2048tables.online/');
  let opened, sent; const popup = {postMessage:(...args)=>{sent=args;}};
  win.open=url=>{opened=new URL(url);return popup;};
  const detail = {hex:'101022109830edba',fullPattern:'free10_256',context:{kind:'guide',step:1}};
  openSiteTab(TAB_IDS.TRAINER,detail,'tables',win);
  assert.equal(opened.searchParams.get('from'),win.location.origin);
  win.listeners.get('message')({source:popup,origin:opened.origin,data:{token:'unique-token',type:'tables-ready'}});
  assert.deepEqual(sent[0].detail,detail);
  assert.equal(sent[1],opened.origin);
  win.listeners.get('message')({source:popup,origin:opened.origin,data:{token:'unique-token',type:'tables-loaded'}});
  assert.equal(win.listeners.size,0); assert.equal(win.timers.size,0);
});
test('receiver accepts a file once, removes handoff parameters, and releases opener', () => {
  const win = windowMock('https://tables.2048tables.online/?tab=analysis&handoff=abc&from=https%3A%2F%2F2048tables.online');
  const messages=[]; const source=win.opener={postMessage:(...args)=>messages.push(args)}; let result;
  receiveSiteContext((...args)=>result=args,win);
  const detail={analysisFile:new File(['replay'],'example.vrs')};
  win.listeners.get('message')({source,origin:'https://2048tables.online',data:{type:'tables-context',token:'abc',tab:TAB_IDS.ANALYSIS,detail}});
  assert.equal(result[1].analysisFile.name,'example.vrs');
  assert.equal(win.listeners.size,0); assert.equal(win.opener,null);
  assert.equal(new URL(win.history.url).searchParams.has('handoff'),false);
  assert.equal(messages.at(-1)[0].type,'tables-loaded');
});
test('existing synchronized settings are not expanded with domain-local gameplay controls', () => {
  const source=readFileSync(new URL('../src/services/preferences/accountPreferences.js',import.meta.url),'utf8');
  const keys=source.split('ACCOUNT_GLOBAL_KEYS = Object.freeze([')[1].split(']);')[0];
  assert.ok(!keys.includes('demo_speed') && !keys.includes('4_spawn_rate'));
});
test('analysis tab is retained when hidden; page and dialog share the same implementation', () => {
  const app=readFileSync(new URL('../src/App.vue',import.meta.url),'utf8');
  assert.match(app,/v-if="isTabOpen\(TAB_IDS.ANALYSIS\)" v-show="activeTab === TAB_IDS.ANALYSIS"/);
  assert.match(app,/:open="true" embedded/);
  const dialog=readFileSync(new URL('../src/features/replay/components/ReplayAnalysisDialog.vue',import.meta.url),'utf8');
  assert.match(dialog,/:disabled="embedded"/);
  assert.match(dialog,/isRunning.value && Object.keys\(nextContext/);
});
