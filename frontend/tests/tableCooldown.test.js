import test from 'node:test';
import assert from 'node:assert/strict';
import { TableDispatcher } from '../src/features/gamer/engine/tableDispatcher.js';

const board = code => [...code].map(c => c === '0' ? 0 : 2 ** parseInt(c, 16));
const current = board('39107a1266d813b1');
const table = (pattern, target, n, free) => ({
  pattern, target, fullPattern: `${pattern}_${target}`, spawnRate: .1,
  ai: { compatible: true, policy_version: 1, large_tiles: n, free_tiles: free },
});
const tables = [table('free10',128,6,6), table('ordinary',512,4,0),
  table('plusone',1024,4,0), table('free12',2048,4,4),
  table('missing',4096,1,0), table('free10',512,6,6)];
const names = d => d.candidates().map(c => c.table.fullPattern);
function setup() {
  const d = new TableDispatcher(tables);
  d.reset(current);
  assert.equal(d.accept({ table: tables[0], type: 1 }, { results: { down: 1 }, dtype: 'uint32' }), 'AI');
  return d;
}

test('handoff excludes lvl+1 and lvl+2 without suppressing ordinary or missing-level candidates', async () => {
  const d = setup();
  assert.ok(names(d).includes('ordinary_512'));
  assert.ok(names(d).includes('missing_4096'));
  for (const name of ['free10_128','plusone_1024','free12_2048']) assert.ok(!names(d).includes(name));
  const requests = [];
  await d.choose(async candidate => {
    requests.push(candidate.table.fullPattern);
    return { results: { down: null }, dtype: 'uint32' };
  });
  assert.ok(requests.length > 0);
  assert.ok(!requests.includes('plusone_1024') && !requests.includes('free12_2048'));
});

test('existing cooldown boundary restores higher-target candidates on twentieth reset', () => {
  const d = setup();
  for (let i=0;i<19;i++) {
    d.reset(current);
    assert.ok(!names(d).includes('free12_2048'));
  }
  d.reset(current);
  assert.ok(names(d).includes('free12_2048'));
  assert.ok(names(d).includes('plusone_1024'));
});

test('another active cooldown extends the restriction, but a normal route to the same table remains eligible', () => {
  const d = setup();
  d.cooldowns.set('other',25);
  for (let i=0;i<20;i++) d.reset(current);
  assert.ok(!names(d).includes('free12_2048'));
  d.reset(board('1011178729ab1cde'));
  assert.ok(names(d).includes('free10_512'));
  assert.ok(names(d).includes('free12_2048'));
});
