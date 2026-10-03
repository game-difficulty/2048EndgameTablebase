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
      const contentKeys = [item.titleKey,item.summaryKey,item.bodyKey,item.noticeKey,...(item.bodyKeys||[]),...(item.featureKeys||[])].filter(Boolean);
      for (const key of contentKeys) assert.equal(typeof value(key), 'string');
      if (item.rewardCopyKey) for (const part of ['title','note','recharge','lucky','envelope','weekly','trophy']) assert.equal(typeof value(`${item.rewardCopyKey}.${part}`), 'string');
      assert.equal(new URL(item.siteUrl || item.liveUrl).protocol, 'https:');
      for (const link of item.links || []) {
        assert.equal(typeof value(link.labelKey), 'string');
        assert.equal(new URL(link.url).protocol, 'https:');
      }
    }
  }
});
test('October update is newest, preserves both earlier notices and resurfaces after an older dismissal', () => {
  assert.equal(latestAnnouncement.id, '2026-10-03-tables-competition-analysis');
  assert.equal(ANNOUNCEMENTS.length, 3);
  assert.equal(findAnnouncement('2026-09-29-play-beta').date, '2026-09-29');
  assert.equal(findAnnouncement('2026-09-16-live-rewards').date, '2026-09-16');
  const storage = { getItem: () => '2026-09-29-play-beta' };
  assert.equal(wasDismissed(storage, latestAnnouncement.id), false);
  assert.equal(latestAnnouncement.links.length, 6);
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
