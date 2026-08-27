import assert from 'node:assert/strict';
import test from 'node:test';

import {
  BaseMinigameEngine,
  trophyLevelForExponent,
  trophyLevelName,
} from '../src/features/minigames/engine/baseEngine.js';
import {
  BlitzkriegEngine,
  DesignMasterEngine,
  EndlessFamilyEngine,
} from '../src/features/minigames/engine/games/index.js';

const definition = (legacyName = 'Classic') => ({
  id: legacyName.toLowerCase().replaceAll(' ', '-'),
  title: legacyName,
  legacyName,
});

test('tile trophy levels advance from bronze through grand', () => {
  assert.equal(trophyLevelForExponent(10, 12), 1);
  assert.equal(trophyLevelForExponent(11, 12), 2);
  assert.equal(trophyLevelForExponent(12, 12), 3);
  assert.equal(trophyLevelForExponent(13, 12), 4);
  assert.equal(trophyLevelName(4), 'grand');
});

test('base engine awards a Grand trophy above the gold exponent', () => {
  const engine = new BaseMinigameEngine(definition(), 0, null, { deferSetup: true });
  engine.board = [
    [13, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  engine.currentMaxNum = 12;
  engine.maxNum = 12;
  engine.isPassed = 3;

  engine.checkGamePassed();

  assert.equal(engine.isPassed, 4);
  assert.deepEqual(engine.popMessages().trophy, {
    level: 'grand',
    message: 'You achieved 8192! You get a grand trophy!',
  });
});

test('a higher Grand tile does not re-award the same trophy tier', () => {
  const engine = new BaseMinigameEngine(definition(), 0, null, { deferSetup: true });
  engine.board = [
    [14, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  engine.currentMaxNum = 13;
  engine.maxNum = 13;
  engine.isPassed = 4;

  engine.checkGamePassed();

  assert.equal(engine.isPassed, 4);
  assert.deepEqual(engine.popMessages().trophy, {
    level: 'grand',
    message: 'You achieved 16384! Take it further!',
  });
});

test('endless games announce the Grand score tier', () => {
  const engine = new EndlessFamilyEngine(definition('Endless Explosions'), 0);
  engine.score = 300000;
  engine.maxScore = 300000;
  engine.currentLevel = 3;
  engine.isPassed = 3;

  engine.queueScoreTrophy();

  assert.equal(engine.currentLevel, 4);
  assert.equal(engine.isPassed, 4);
  assert.deepEqual(engine.popMessages().trophy, {
    level: 'grand',
    message: 'You achieved 300k score! You get a grand trophy!',
  });
});

test('Design Master awards Grand above its gold pattern exponent', () => {
  const engine = new DesignMasterEngine(definition('Design Master1'), 0);
  engine.maxNum = 10;
  engine.currentMaxNum = 10;
  engine.isPassed = 3;
  engine.checkPattern = () => 11;

  engine.checkGamePassed();

  assert.equal(engine.isPassed, 4);
  assert.deepEqual(engine.popMessages().trophy, {
    level: 'grand',
    message: 'You achieved 2048! You get a grand trophy!',
  });
});

test('Blitzkrieg labels a tile above gold as Grand', () => {
  const engine = new BlitzkriegEngine(definition('Blitzkrieg'), 0);
  engine.board = [
    [13, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ];
  engine.currentMaxNum = 12;
  engine.maxNum = 12;
  engine.isPassed = 3;
  engine.isOver = true;
  engine.score = 1000;
  engine.maxScore = 1000;

  engine.checkGamePassed();

  assert.equal(engine.isPassed, 4);
  assert.deepEqual(engine.popMessages().trophy, {
    level: 'grand',
    message: 'You achieved 1000 score! You get a grand trophy!',
  });
});
