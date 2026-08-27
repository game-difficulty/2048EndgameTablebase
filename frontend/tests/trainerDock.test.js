import assert from 'node:assert/strict';
import test from 'node:test';

import {
  TRAINER_DOCK_PLACEMENTS,
  isTrainerDocked,
  normalizeTrainerDockPlacement,
  resolveTrainerJumpDockPlacement,
} from '../src/app/trainerDock.js';
import { TAB_IDS } from '../src/app/tabRegistry.js';
import { useTabManager } from '../src/app/useTabManager.js';

test('normalizes Trainer dock placements deterministically', () => {
  assert.equal(normalizeTrainerDockPlacement('right'), TRAINER_DOCK_PLACEMENTS.RIGHT);
  assert.equal(normalizeTrainerDockPlacement('bottom'), TRAINER_DOCK_PLACEMENTS.BOTTOM);
  assert.equal(normalizeTrainerDockPlacement('invalid'), TRAINER_DOCK_PLACEMENTS.NONE);
  assert.equal(isTrainerDocked(TRAINER_DOCK_PLACEMENTS.NONE), false);
  assert.equal(isTrainerDocked(TRAINER_DOCK_PLACEMENTS.RIGHT), true);
});

test('guide jumps reuse a valid dock preference and narrow layouts fall back to full view', () => {
  assert.equal(resolveTrainerJumpDockPlacement({
    placement: TRAINER_DOCK_PLACEMENTS.BOTTOM,
    dockAvailable: true,
    sourceIsHelp: true,
  }), TRAINER_DOCK_PLACEMENTS.BOTTOM);
  assert.equal(resolveTrainerJumpDockPlacement({
    placement: TRAINER_DOCK_PLACEMENTS.NONE,
    dockAvailable: true,
    sourceIsHelp: true,
  }), TRAINER_DOCK_PLACEMENTS.RIGHT);
  assert.equal(resolveTrainerJumpDockPlacement({
    placement: TRAINER_DOCK_PLACEMENTS.RIGHT,
    dockAvailable: false,
    sourceIsHelp: true,
  }), TRAINER_DOCK_PLACEMENTS.NONE);
});

test('opening Trainer in the background preserves the companion tab', () => {
  const tabs = useTabManager();
  tabs.openTab(TAB_IDS.HELP);
  tabs.openTabInBackground(TAB_IDS.TRAINER);

  assert.equal(tabs.activeTab.value, TAB_IDS.HELP);
  assert.equal(tabs.isTabOpen(TAB_IDS.TRAINER), true);
});
