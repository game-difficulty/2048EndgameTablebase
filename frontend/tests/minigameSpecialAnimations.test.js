import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildMoveAnimationMetadata,
  computeMoveAnimation,
  mergeLine,
} from '../src/features/minigames/engine/boardMover.js';
import {
  BlitzkriegEngine,
  EndlessFamilyEngine,
  IceAgeEngine,
  MysteryMergeEngine,
} from '../src/features/minigames/engine/games/index.js';
import {
  activatePowerup,
  applyTargetAction,
} from '../src/features/minigames/engine/powerups.js';

const definition = (legacyName, moduleKey) => ({
  id: legacyName.toLowerCase().replaceAll(' ', '-'),
  title: legacyName,
  legacyName,
  moduleKey,
});

const powerupState = (engine, counts = { bomb: 1, glove: 1, twist: 1 }) => ({
  engine,
  difficulty: 0,
  powerupCounts: { ...counts },
  activeMode: null,
  interactionPhase: 0,
  selectionCache: null,
});

test('wall-separated segments merge and animate independently', () => {
  assert.deepEqual(mergeLine([1, 1, -1, 2, 2]).line, [2, 0, -1, 3, 0]);
  assert.deepEqual(mergeLine([-3, -3, -1, 1, 1]).line, [-3, 0, -1, 2, 0]);

  const animation = computeMoveAnimation([[1, 1, -1, 2, 2]], 'left');
  assert.deepEqual(animation.slide_distances, [0, 1, 0, 0, 1]);
  assert.deepEqual(animation.pop_positions, [1, 0, 0, 1, 0]);
});

test('zero-valued independent spawns are retained in animation metadata', () => {
  const metadata = buildMoveAnimationMetadata([[1, 0], [0, 0]], 'left', 3, 0);
  assert.deepEqual(metadata.appearTile, { index: 3, value: 0 });
});

test('Mystery Merge powerups keep the hidden mask attached to tiles', () => {
  const engine = new MysteryMergeEngine(definition('Mystery Merge2', 'mystery_merge'), 0);
  engine.board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  engine.masked = [[true, false, false, false], [false, false, false, false], [false, false, false, false], [false, false, false, false]];
  const state = powerupState(engine);

  assert.equal(activatePowerup(state, 'glove'), true);
  assert.equal(applyTargetAction(state, 0), false);
  assert.equal(applyTargetAction(state, 1), true);
  assert.equal(engine.masked[0][0], false);
  assert.equal(engine.masked[0][1], true);
});

test('Ice Age powerups clear stale countdown overlays', () => {
  const engine = new IceAgeEngine(definition('Ice Age', 'ice_age'), 0);
  engine.countDown = [[10, 20, 0, 0], [30, 40, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];

  engine.applyPowerupTwist(0, 0);
  assert.deepEqual(engine.countDown.slice(0, 2).map((row) => row.slice(0, 2)), [[0, 0], [0, 0]]);
});

test('Endless independent objects rotate with an otherwise empty 2x2 region', () => {
  const engine = new EndlessFamilyEngine(definition('Endless Explosions', 'endless_family'), 0);
  engine.board = Array.from({ length: 4 }, () => new Array(4).fill(0));
  engine.bombPos = [0, 0];
  const state = powerupState(engine);

  assert.equal(activatePowerup(state, 'twist'), true);
  assert.equal(applyTargetAction(state, 0), true);
  assert.deepEqual(engine.bombPos, [0, 1]);
  assert.equal(engine.animation.effects[0].tiles.some((tile) => tile.kind === 'independent_object'), true);
});

test('bomb effects consume the tile after the burst starts', () => {
  const engine = new MysteryMergeEngine(definition('Mystery Merge2', 'mystery_merge'), 0);
  engine.board = [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  engine.masked[0][0] = true;
  const state = powerupState(engine);

  assert.equal(activatePowerup(state, 'bomb'), true);
  assert.equal(applyTargetAction(state, 0), true);
  assert.equal(engine.masked[0][0], false);
  assert.deepEqual(engine.animation.effects[0].consumeIndices, [0]);
  assert.equal(engine.animation.effects[0].consumeDelayMs, 80);
});

const blockedBoard = () => [
  [1, 2, 1, 2],
  [2, 1, 2, 1],
  [1, 2, 1, 2],
  [2, 1, 2, 1],
];

test('Mystery Merge resumes after a powerup restores a legal move', () => {
  const engine = new MysteryMergeEngine(definition('Mystery Merge2', 'mystery_merge'), 0);
  engine.board = blockedBoard();
  engine.masked = Array.from({ length: 4 }, () => new Array(4).fill(false));
  engine.checkGameOver();
  assert.equal(engine.isOver, true);
  assert.equal(engine.revealAll, true);

  const state = powerupState(engine, { bomb: 1, glove: 0, twist: 0 });
  assert.equal(activatePowerup(state, 'bomb'), true);
  assert.equal(applyTargetAction(state, 0), true);
  assert.equal(engine.isOver, false);
  assert.equal(engine.revealAll, false);
});

test('Blitzkrieg resumes after a powerup restores a legal move while time remains', () => {
  const engine = new BlitzkriegEngine(definition('Blitzkrieg', 'blitzkrieg'), 0);
  engine.board = blockedBoard();
  engine.remainingMs = 60_000;
  engine.timerRunning = false;
  engine.timerAnchorMs = null;
  engine.checkGameOver();
  assert.equal(engine.isOver, true);

  const state = powerupState(engine, { bomb: 1, glove: 0, twist: 0 });
  assert.equal(activatePowerup(state, 'bomb'), true);
  assert.equal(applyTargetAction(state, 0), true);
  assert.equal(engine.isOver, false);
});

test('Blitzkrieg cannot resume after its timer expires', () => {
  const engine = new BlitzkriegEngine(definition('Blitzkrieg', 'blitzkrieg'), 0);
  engine.board = blockedBoard();
  engine.remainingMs = 0;
  engine.isOver = true;

  const state = powerupState(engine, { bomb: 1, glove: 0, twist: 0 });
  assert.equal(activatePowerup(state, 'bomb'), true);
  assert.equal(applyTargetAction(state, 0), true);
  assert.equal(engine.isOver, true);
});
