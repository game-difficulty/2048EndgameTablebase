import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ANNOUNCEMENTS, latestAnnouncement, findAnnouncement, resolveAnnouncementTarget, wasDismissed, dismissAnnouncement } from '../src/features/announcements/catalog.js';
import { TAB_IDS } from '../src/app/tabRegistry.js';
import { useTabManager } from '../src/app/useTabManager.js';

test('catalog has unique stable IDs and all bilingual content', () => {
  assert.equal(new Set(ANNOUNCEMENTS.map(item=>item.id)).size, ANNOUNCEMENTS.length);
  for (const language of ['zh', 'en']) {
    const messages = JSON.parse(readFileSync(new URL(`../src/locales/${language}.json`, import.meta.url), 'utf8'));
    const value = key => key.split('.').reduce((obj, part)=>obj?.[part], messages);
    for (const item of ANNOUNCEMENTS) {
      for (const key of [item.titleKey,item.summaryKey,item.bodyKey]) assert.equal(typeof value(key), 'string');
      for (const part of ['title','note','recharge','lucky','envelope','weekly','trophy']) assert.equal(typeof value(`${item.rewardCopyKey}.${part}`), 'string');
      assert.equal(new URL(item.liveUrl).protocol, 'https:');
    }
  }
});
test('dismiss only current notice; blocked storage does not break navigation', () => {
  const values = new Map();
  const storage = {getItem:key=>values.get(key),setItem:(key,value)=>values.set(key,value)};
  assert.equal(wasDismissed(storage,latestAnnouncement.id),false);
  dismissAnnouncement(storage,latestAnnouncement.id);
  assert.equal(wasDismissed(storage,latestAnnouncement.id),true);
  assert.equal(wasDismissed(storage,'future'),false);
  const blocked = {getItem(){throw Error();},setItem(){throw Error();}};
  assert.equal(wasDismissed(blocked,latestAnnouncement.id),false);
  assert.doesNotThrow(()=>dismissAnnouncement(blocked,latestAnnouncement.id));
});
test('configured targets, unknown IDs and reusable closable tab', () => {
  const target = resolveAnnouncementTarget(latestAnnouncement.target);
  assert.equal(target.tab,TAB_IDS.ANNOUNCEMENTS);
  assert.equal(findAnnouncement('missing').id,latestAnnouncement.id);
  assert.equal(resolveAnnouncementTarget({type:'tab',id:TAB_IDS.GAMER}).tab,TAB_IDS.GAMER);
  assert.equal(resolveAnnouncementTarget({type:'tab',id:'javascript:bad'}).tab,TAB_IDS.ANNOUNCEMENTS);
  const tabs = useTabManager();
  tabs.openTab(target.tab); tabs.openTab(target.tab);
  assert.equal(tabs.openTabs.value.filter(id=>id===target.tab).length,1);
  tabs.closeTab(target.tab);
  assert.equal(tabs.activeTab.value,TAB_IDS.MAIN_MENU);
});
