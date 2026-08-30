import assert from 'node:assert/strict';
import test from 'node:test';

import {
  TRAINER_DOCK_LAYOUT,
  TRAINER_DOCK_PLACEMENTS,
  isTrainerDocked,
  normalizeTrainerDockPlacement,
  resolveTrainerJumpDockPlacement,
  resolveTrainerDockSurfaceHeight,
} from '../src/app/trainerDock.js';
import {
  KEYBOARD_OWNERS,
  keyboardInputAllowed,
  resetKeyboardOwnership,
  setKeyboardOwner,
  setSplitKeyboardMode,
} from '../src/app/keyboardOwnership.js';
import { TAB_IDS } from '../src/app/tabRegistry.js';
import { useTabManager } from '../src/app/useTabManager.js';

test.afterEach(() => resetKeyboardOwnership());

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

test('battle practice jumps prefer the Trainer dock without taking a full tab', () => {
  assert.equal(resolveTrainerJumpDockPlacement({
    placement: TRAINER_DOCK_PLACEMENTS.NONE,
    dockAvailable: true,
    preferDock: true,
  }), TRAINER_DOCK_PLACEMENTS.RIGHT);
});

test('opening Trainer in the background preserves the companion tab', () => {
  const tabs = useTabManager();
  tabs.openTab(TAB_IDS.HELP);
  tabs.openTabInBackground(TAB_IDS.TRAINER);

  assert.equal(tabs.activeTab.value, TAB_IDS.HELP);
  assert.equal(tabs.isTabOpen(TAB_IDS.TRAINER), true);
});

test('bottom docking adds a second full-height content page', () => {
  assert.equal(resolveTrainerDockSurfaceHeight({
    placement: TRAINER_DOCK_PLACEMENTS.BOTTOM,
    baseHeight: 800,
    topBarHeight: 48,
  }), 1552);
  assert.equal(resolveTrainerDockSurfaceHeight({
    placement: TRAINER_DOCK_PLACEMENTS.RIGHT,
    baseHeight: 800,
    topBarHeight: 48,
  }), 800);
});

test('right docking uses the ten-percent narrower layout dimensions', () => {
  assert.equal(TRAINER_DOCK_LAYOUT.RIGHT_WIDTH_PX, 450);
  assert.equal(TRAINER_DOCK_LAYOUT.RIGHT_BOARD_WIDTH_PX, 324);
});

test('docked Trainer and primary page never share keyboard ownership', () => {
  setSplitKeyboardMode(true);
  setKeyboardOwner(KEYBOARD_OWNERS.TRAINER);
  assert.equal(keyboardInputAllowed(KEYBOARD_OWNERS.TRAINER), true);
  assert.equal(keyboardInputAllowed(KEYBOARD_OWNERS.PRIMARY), false);

  setKeyboardOwner(KEYBOARD_OWNERS.PRIMARY);
  assert.equal(keyboardInputAllowed(KEYBOARD_OWNERS.TRAINER), false);
  assert.equal(keyboardInputAllowed(KEYBOARD_OWNERS.PRIMARY), true);
});
