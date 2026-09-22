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
  IceAgeEngine,
} from '../src/features/minigames/engine/games/index.js';
import {
  MINIGAME_RULE_IDS,
  MINIGAME_TROPHY_THRESHOLDS,
  formatMinigameRules,
  getMinigameRuleView,
} from '../src/features/minigames/model/minigameRules.js';
import { MINIGAME_REGISTRY } from '../src/features/minigames/engine/registry.js';

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
  engine.score = 600000;
  engine.maxScore = 600000;
  engine.currentLevel = 3;
  engine.isPassed = 3;

  engine.queueScoreTrophy();

  assert.equal(engine.currentLevel, 4);
  assert.equal(engine.isPassed, 4);
  assert.deepEqual(engine.popMessages().trophy, {
    level: 'grand',
    message: 'You achieved 600k score! You get a grand trophy!',
  });
});

test('endless and hybrid games use the raised cumulative-score thresholds', () => {
  assert.deepEqual(MINIGAME_TROPHY_THRESHOLDS.endless, [50000, 120000, 300000, 600000]);
  assert.deepEqual(MINIGAME_TROPHY_THRESHOLDS.hybrid, [30000, 60000, 120000, 240000]);

  const standard = new EndlessFamilyEngine(definition('Endless AirRaid'), 0);
  assert.deepEqual(standard.levels, [[600000, 4], [300000, 3], [120000, 2], [50000, 1]]);
  const hybrid = new EndlessFamilyEngine(definition('Endless Hybrid'), 1);
  assert.deepEqual(hybrid.levels, [[240000, 4], [120000, 3], [60000, 2], [30000, 1]]);
});

test('an existing endless trophy is retained and only a higher tier is announced', () => {
  const engine = new EndlessFamilyEngine(definition('Endless Explosions'), 0);
  engine.isPassed = 2;
  engine.currentLevel = 0;
  engine.score = 50000;
  engine.maxScore = 50000;

  engine.queueScoreTrophy();
  assert.equal(engine.isPassed, 2);
  assert.equal(engine.currentLevel, 0);
  assert.deepEqual(engine.popMessages(), {});

  engine.score = 300000;
  engine.maxScore = 300000;
  engine.queueScoreTrophy();
  assert.equal(engine.isPassed, 3);
  assert.equal(engine.currentLevel, 3);
  assert.equal(engine.popMessages().trophy.level, 'gold');
});

test('Ice Age freezes later on Casual and earlier on Hard', () => {
  assert.equal(new IceAgeEngine(definition('Ice Age'), 0).frozenStep, 100);
  assert.equal(new IceAgeEngine(definition('Ice Age'), 1).frozenStep, 80);
});

test('rule copy is bilingual and exposes exact current trophy targets', () => {
  const zh = getMinigameRuleView('endless-hybrid', 1, 'zh-CN');
  const en = getMinigameRuleView('endless-hybrid', 1, 'en');
  assert.match(zh.summary, /混合/);
  assert.match(en.summary, /endless mix/i);
  assert.deepEqual(zh.trophies.map((row) => row.requirement), ['≥ 30,000', '≥ 60,000', '≥ 120,000', '≥ 240,000']);
  assert.match(formatMinigameRules('ice-age', 0, 'zh-CN'), /100/);
  assert.match(formatMinigameRules('ice-age', 1, 'en'), /80/);
});

test('every registered minigame has a dedicated bilingual rules entry', () => {
  assert.equal(MINIGAME_REGISTRY.length, 20);
  assert.deepEqual(
    [...MINIGAME_RULE_IDS].sort(),
    MINIGAME_REGISTRY.map(({ id }) => id).sort(),
  );
  for (const { id } of MINIGAME_REGISTRY) {
    const zh = getMinigameRuleView(id, 0, 'zh-CN');
    const en = getMinigameRuleView(id, 1, 'en');
    assert.ok(zh.summary && zh.objective && zh.mechanics.length, `${id} is missing Chinese rules`);
    assert.ok(en.summary && en.objective && en.mechanics.length, `${id} is missing English rules`);
  }
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
