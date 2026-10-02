import test from 'node:test';
import assert from 'node:assert/strict';
import { PracticeSpecialGame, slideSpecialTiles } from '../src/projects/practiceSpecialEngine.js';
import { ticketFloat } from '../src/projects/randomStreams.js';

const number = (id, value, cell) => ({ id, kind: 'number', value, cells: [cell] });
const special = (id, kind, cell) => ({ id, kind, value: 0, cells: [cell] });

test('ordinary numeric merges still score like 2048', () => {
  const result = slideSpecialTiles([number('a', 2, 0), number('b', 2, 1)], 'left');
  assert.equal(result.score, 4);
  assert.deepEqual(result.tiles.map(tile => [tile.value, tile.cells]), [[4, [0]]]);
  assert.equal(result.tiles[0].merged, undefined);
});

test('new adjacent special pair replaces old domino and moves rigidly', () => {
  const project = { id: 'pair', rows: 4, cols: 4, specialRule: 'pair', specialSpawnRate: 0 };
  const game = new PracticeSpecialGame(project, { seed: 'fixed' });
  game.tiles = [
    { id: 'old', kind: 'pair-double', value: 0, cells: [12, 13] },
    special('first', 'pair-single', 1), special('second', 'pair-single', 2),
  ];
  game.react({ merges: [], removals: [], changed: false });
  assert.deepEqual(game.tiles.map(tile => tile.cells), [[1, 2]]);
  const moved = slideSpecialTiles(game.tiles, 'down');
  assert.deepEqual(moved.tiles[0].cells, [13, 14]);
});

test('same-color chemicals disappear; unlike chemicals merge into one wall at the final cell', () => {
  const collision = slideSpecialTiles([special('a', 'chemical-a', 0), special('b', 'chemical-a', 2)], 'left');
  assert.equal(collision.tiles.length, 0);
  assert.equal(collision.score, 0);
  const game = new PracticeSpecialGame({ id: 'chem', rows: 4, cols: 4, specialRule: 'chemical', specialSpawnRate: 0 });
  game.tiles = [special('a', 'chemical-a', 4), special('b', 'chemical-b', 5)];
  game.react({ merges: [], removals: [], changed: false });
  assert.deepEqual(game.tiles.map(tile => tile.kind), ['chemical-a', 'chemical-b']);
  const parallel = slideSpecialTiles(game.tiles, 'down');
  assert.deepEqual(parallel.tiles.map(tile => [tile.kind, tile.cells]), [
    ['chemical-a', [12]], ['chemical-b', [13]],
  ]);
  const merged = slideSpecialTiles(game.tiles, 'right');
  assert.deepEqual(merged.tiles.map(tile => [tile.kind, tile.cells]), [['wall', [7]]]);
  assert.equal(merged.merges.length, 1);
  assert.deepEqual(merged.merges[0].sources, ['b', 'a']);
  assert.equal(merged.score, 0);
});

test('special spawn chance loses two percentage points per active special', () => {
  const examples = [
    { rule: 'pair', kinds: ['pair-single', 'pair-single', 'pair-single'], ignored: { id: 'ordinary', kind: 'number', value: 2, cells: [12] } },
    { rule: 'chemical', kinds: ['chemical-a', 'chemical-b', 'chemical-a'] },
    { rule: 'bomb', kinds: ['bomb', 'bomb', 'bomb'] },
  ];
  for (const { rule, kinds, ignored } of examples) {
    const game = new PracticeSpecialGame({ id: rule, rows: 4, cols: 4, specialRule: rule, specialSpawnRate: .05 }, { seed: 'fixed' });
    const extras = ignored ? [ignored] : [];
    for (const [count, chance] of [.05, .03, .01, 0].entries()) {
      game.tiles = [...extras, ...kinds.slice(0, count).map((kind, index) => special(`s${index}`, kind, index))];
      assert.ok(Math.abs(game.specialSpawnChance() - chance) < 1e-10, `${rule}: ${count} specials`);
    }
  }
});

test('a bonded domino counts as one special for spawn probability', () => {
  const game = new PracticeSpecialGame({ id:'pair', specialRule:'pair', specialSpawnRate:.05 }, { seed:'domino-count' });
  const domino = { id:'double', kind:'pair-double', value:0, cells:[12,13] };
  game.tiles = [domino];
  assert.ok(Math.abs(game.specialSpawnChance() - .03) < 1e-10);
  game.tiles.push(special('single', 'pair-single', 0));
  assert.ok(Math.abs(game.specialSpawnChance() - .01) < 1e-10);
  game.tiles.push(special('another', 'pair-single', 2));
  assert.equal(game.specialSpawnChance(), 0);
  game.tiles = [domino];
  const ticket = Array.from({length:10000}, (_,i)=>i).find(i=>ticketFloat(i,'special')>=.03 && ticketFloat(i,'special')<.05);
  assert.notEqual(ticket, undefined);
  let draws = 0;
  game.nextTicket = () => { draws++; return ticket; };
  assert.equal(game.spawn().kind, 'number');
  assert.equal(draws, 1);
});

test('spawn applies reduced chance to the fixed ticket without consuming extra randomness', () => {
  const ticket = Array.from({ length: 10000 }, (_, index) => index)
    .find(value => ticketFloat(value, 'special') >= .01 && ticketFloat(value, 'special') < .03);
  assert.notEqual(ticket, undefined);
  for (const [rule, kind] of [['pair', 'pair-single'], ['chemical', 'chemical-a'], ['bomb', 'bomb']]) {
    const game = new PracticeSpecialGame({ id: rule, rows: 4, cols: 4, specialRule: rule, specialSpawnRate: .05 }, { seed: 'fixed' });
    game.nextTicket = () => ticket;
    game.tiles = [special('one', kind, 0)];
    assert.notEqual(game.spawn().kind, 'number', `${rule}: one special permits this ticket`);
    game.tiles = [special('one', kind, 0), special('two', kind, 1)];
    assert.equal(game.spawn().kind, 'number', `${rule}: two specials reject this ticket`);
  }
});

test('bomb countdown changes only on movement and merges by sum minus one', () => {
  const bomb = (id, cell, countdown) => ({ id, kind: 'bomb', value: 0, cells: [cell], countdown, baseCountdown: countdown });
  const still = slideSpecialTiles([bomb('one', 0, 1)], 'left');
  assert.equal(still.changed, false);
  assert.equal(still.tiles[0].countdown, 1);
  const moved = slideSpecialTiles([bomb('one', 0, 1)], 'right');
  assert.equal(moved.tiles[0].kind, 'wall');
  assert.deepEqual(moved.tiles[0].cells, [3]);
  const merged = slideSpecialTiles([bomb('a', 0, 3), bomb('b', 2, 4)], 'left');
  assert.equal(merged.tiles[0].kind, 'bomb');
  assert.equal(merged.tiles[0].countdown, 6);
});

test('a fixed seed reproduces special spawn sequence', () => {
  const project = { id: 'bomb', rows: 4, cols: 4, specialRule: 'bomb', specialSpawnRate: .35,
    bombCountdownMin: 12, bombCountdownMax: 32 };
  const a = new PracticeSpecialGame(project, { seed: 'same' });
  const b = new PracticeSpecialGame(project, { seed: 'same' });
  assert.deepEqual(a.snapshot().tiles, b.snapshot().tiles);
  for (const direction of ['left', 'down', 'right', 'up', 'left']) {
    a.move(direction); b.move(direction);
    assert.deepEqual(a.snapshot().tiles, b.snapshot().tiles);
  }
});

test('chemical color order stays shared when board-dependent spawn timings differ', () => {
  const project = { id: 'chemical', rows: 4, cols: 4, specialRule: 'chemical', specialSpawnRate: .05 };
  const a = new PracticeSpecialGame(project, { seed: 'shared-colors' });
  const b = new PracticeSpecialGame(project, { seed: 'shared-colors' });
  const colorsA = [], colorsB = [], stepsA = [], stepsB = [];
  for (let step = 0; step < 10000; step++) {
    a.tiles = [];
    b.tiles = [special('existing-a', 'chemical-a', 0), special('existing-b', 'chemical-b', 1)];
    for (const [game, colors, steps] of [[a, colorsA, stepsA], [b, colorsB, stepsB]]) {
      const spawned = game.spawn();
      if (spawned.kind !== 'number') { colors.push(spawned.kind); steps.push(step); }
    }
  }
  assert.ok(colorsB.length > 50);
  assert.ok(colorsA.length > colorsB.length);
  assert.notDeepEqual(stepsA.slice(0, stepsB.length), stepsB);
  assert.deepEqual(colorsA.slice(0, colorsB.length), colorsB);
  assert.equal(a.randomState, b.randomState, 'colors do not consume numeric spawn tickets');
  assert.equal(new Set(colorsA).size, 2);
});

test('ordinary spawns, blocked spawns and ineffective moves do not advance chemical colors; restart continues the sequence', () => {
  const project = { id: 'chemical', rows: 4, cols: 4, specialRule: 'chemical', specialSpawnRate: 0 };
  const game = new PracticeSpecialGame(project, { seed: 'colors-only-on-spawn' });
  const state = game.chemicalColorState;
  game.tiles = [];
  assert.equal(game.spawn().kind, 'number');
  game.tiles = [number('corner', 2, 0)];
  assert.equal(game.move('left').changed, false);
  game.tiles = Array.from({ length: 16 }, (_, cell) => number(String(cell), 2, cell));
  assert.equal(game.spawn(), null);
  game.reset();
  assert.equal(game.chemicalColorState, state);
  game.project = { ...project, specialSpawnRate: 1 };
  game.tiles = [];
  assert.ok(game.spawn().kind.startsWith('chemical-'));
  const advanced = game.chemicalColorState;
  assert.notEqual(advanced, state);
  game.project = project;
  game.reset();
  assert.equal(game.chemicalColorState, advanced);
});

test('spawned bombs get varied integer countdowns within the inclusive 12–32 range', () => {
  const project = { id: 'bomb-range', rows: 4, cols: 4, specialRule: 'bomb', specialSpawnRate: 1,
    bombCountdownMin: 12, bombCountdownMax: 32 };
  const game = new PracticeSpecialGame(project, { seed: 'countdowns' });
  const countdowns = [];
  for (let index = 0; index < 300; index += 1) {
    game.tiles = [];
    const spawned = game.spawn();
    assert.equal(spawned.kind, 'bomb');
    assert.equal(Number.isInteger(spawned.countdown), true);
    assert.ok(spawned.countdown >= 12 && spawned.countdown <= 32);
    countdowns.push(spawned.countdown);
  }
  assert.ok(countdowns.includes(12));
  assert.ok(countdowns.includes(32));
  assert.ok(new Set(countdowns).size > 1);
});
