import test from 'node:test';
import assert from 'node:assert/strict';
import { AUXILIARY_ENTRIES, AUXILIARY_GROUPS, HOME_AUXILIARY_ENTRIES } from '../src/app/auxiliaryEntries.js';
import { TAB_IDS, TAB_REGISTRY } from '../src/app/tabRegistry.js';
import { useTabManager } from '../src/app/useTabManager.js';

test('external entries stay outside the internal tab lifecycle', () => {
  assert.equal(AUXILIARY_ENTRIES.live.href, 'https://live.2048tables.online/');
  assert.equal(AUXILIARY_ENTRIES.replay.href, '/verse-replay/');
  assert.equal(AUXILIARY_ENTRIES.live.tab, undefined);
  assert.equal(AUXILIARY_ENTRIES.replay.tab, undefined);
});
test('homepage and directory share the same entry definitions', () => {
  assert.deepEqual(HOME_AUXILIARY_ENTRIES.map(e => e.id), ['announcements', 'replay', 'more', 'github']);
  assert.deepEqual(AUXILIARY_GROUPS.flatMap(g => g.entries.map(e => e.id)), ['live', 'replay', 'announcements', 'quota', 'help']);
  for (const entry of Object.values(AUXILIARY_ENTRIES)) {
    assert.equal([entry.href, entry.tab, entry.dialog].filter(Boolean).length, 1);
    if (entry.tab) assert.ok(TAB_REGISTRY[entry.tab]);
  }
});
test('more page opens, closes and never displaces the permanent homepage', () => {
  const tabs = useTabManager();
  tabs.openTab(TAB_IDS.MORE);
  assert.equal(tabs.activeTab.value, TAB_IDS.MORE);
  tabs.closeTab(TAB_IDS.MORE);
  assert.equal(tabs.activeTab.value, TAB_IDS.MAIN_MENU);
  tabs.closeTab(TAB_IDS.MAIN_MENU);
  assert.deepEqual(tabs.openTabs.value, [TAB_IDS.MAIN_MENU]);
});
