import assert from 'node:assert/strict';
import test from 'node:test';
import { compileTableStructure, matchTableStructure } from '../src/features/gamer/engine/tableStructure.js';
import { TableDispatcher, maskLargeTiles } from '../src/features/gamer/engine/tableDispatcher.js';
import { TableAiCache } from '../src/features/gamer/services/tableAiCache.js';

const metadata = (masks) => ({ version: 1, transforms: 'dihedral8', pattern_masks: masks });
const values = (hex) => [...hex].map((digit) => parseInt(digit, 16)).map((code) => code ? 2 ** code : 0);
const initial = values('cb10a94173303220');
const tables = [
  ['4431', '1024', '00000000000f0fff'],
  ['444', '1024', '000000000000ffff'],
  ['LL', '1024', '0000000000ff00ff'],
  ['444', '2048', '000000000000ffff'],
].map(([pattern, target, mask]) => ({ pattern, target, fullPattern: `${pattern}_${target}`, spawnRate: .1,
  ai: { compatible: true, policy_version: 1, large_tiles: 4, free_tiles: 0, structure: metadata([mask]) } }));

test('reference board skips only structurally impossible candidates before lookup', async () => {
  const dispatcher = new TableDispatcher(tables);
  dispatcher.reset(initial);
  const calls = [];
  assert.equal(await dispatcher.choose(async (candidate, hex) => {
    calls.push([candidate.table.fullPattern, hex]);
    return { results: { up: .994990912 }, dtype: 'uint32' };
  }), 'up');
  assert.deepEqual(calls, [['LL_1024', 'ff10ff4173303220']]);
  assert.deepEqual([...dispatcher.cooldowns], []);
  assert.deepEqual(dispatcher.board, initial);
});

test('unknown, old or malformed metadata queries candidates in the original order', async () => {
  for (const structure of [undefined, { ...metadata([]), version: 2 }, metadata(['xyz']),
    { ...metadata([]), transforms: 'rotations4' }, metadata([123])]) {
    const available = tables.map((table) => ({ ...table, ai: { ...table.ai, structure } }));
    const dispatcher = new TableDispatcher();
    dispatcher.setTables(available, .1, available.map(table => table.fullPattern));
    dispatcher.reset(initial);
    const calls = [];
    const direction = await dispatcher.choose(async ({ table }) => {
      calls.push(table.fullPattern);
      return { results: table.pattern === 'LL' ? { up: .99 } : {} };
    });
    assert.equal(direction, 'up');
    assert.deepEqual(calls, ['4431_1024', '444_1024', 'LL_1024']);
  }
});

test('no matching structure falls directly back to local AI without a request', async () => {
  const dispatcher = new TableDispatcher(tables.filter((table) => table.pattern !== 'LL'));
  dispatcher.reset(initial);
  assert.equal(await dispatcher.choose(() => assert.fail('Unexpected lookup')), 'AI');
});

test('empty masks accept any structure and bit masks retain native bitwise semantics', () => {
  const board = values('123456789abcdef0');
  assert.equal(matchTableStructure(board, board, 0, compileTableStructure(metadata([]))), 'match');
  assert.equal(matchTableStructure(board, board, 0,
    compileTableStructure(metadata(['00000000000000e0']))), 'match', 'f satisfies an e mask');
  assert.equal(matchTableStructure(board, board, 0,
    compileTableStructure(metadata(['ffffffffffffffff']))), 'mismatch');
});

test('ambiguous large-tile masking never causes a false rejection', () => {
  const rules = compileTableStructure(metadata(['ffffffffffffffff']));
  const check = (board, count) => matchTableStructure(board, maskLargeTiles(board, count), count, rules);
  const tied = [1024, 512, 512, 128, ...Array(12).fill(2)];
  assert.equal(check(tied, 2), 'unknown');
  assert.equal(check([65536, 32768, 32768, ...Array(13).fill(2)], 2), 'mismatch');
  assert.equal(check([131072, 65536, ...Array(14).fill(2)], 1), 'unknown');
  assert.equal(check([131072, 65536, ...Array(14).fill(2)], 2), 'mismatch');
  assert.equal(check([65536, 65536, ...Array(14).fill(2)], 1), 'unknown');
});

test('catalog replacement recompiles structure rules', async () => {
  const dispatcher = new TableDispatcher(tables);
  dispatcher.reset(initial);
  dispatcher.setTables(tables.map((table) => ({ ...table, ai: { ...table.ai, structure: metadata([]) } })), .1);
  const calls = [];
  await dispatcher.choose(async ({ table }) => {
    calls.push(table.fullPattern);
    return { results: { left: .8 } };
  });
  assert.deepEqual(calls, ['4431_1024']);
});

test('cached LL route steps are not blocked or aborted by impossible candidate probes', async () => {
  const boards = ['cb10a94173303220', 'cb11a94273303220', 'cb10a94173313222', 'cb10a94174113320'];
  const requests = [];
  const cache = new TableAiCache({ transport: { open(body, { onResult }) {
    requests.push(body);
    assert.equal(body.full_pattern, 'LL_1024');
    boards.forEach((hex, step) => onResult({ ...body, type: 'result', seq: step,
      board_codes: [...hex].map((digit) => parseInt(digit, 16)), rng_state: [step + 1, 2, 3, 4],
      results: { up: .994 }, dtype: 'uint32' }));
    return { credit() {}, cancel() {} };
  }, close() {} } });
  const dispatcher = new TableDispatcher(tables);
  for (let step = 0; step < 3; step += 1) {
    dispatcher.reset(values(boards[step]));
    const probes = [];
    assert.equal(await dispatcher.choose(({ table }) => {
      probes.push(table.fullPattern);
      const body = { catalog_version: 'v1', full_pattern: table.fullPattern,
        board_codes: [...boards[step]].map((digit) => parseInt(digit, 16)),
        rng_state: [step + 1, 2, 3, 4], spawn_rate4: .1, difficulty: 0, random_only: false };
      if (step) assert.ok(cache.get(body), 'Continuation should already be cached');
      return cache.lookup(body);
    }), 'up');
    assert.deepEqual(probes, ['LL_1024']);
    await new Promise((resolve) => setImmediate(resolve));
  }
  assert.equal(requests.length, 1, 'The LL subscription remains open across steps');
  cache.clear();
});
