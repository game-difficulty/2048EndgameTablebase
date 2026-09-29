import test from 'node:test';
import assert from 'node:assert/strict';
import { MatchRuntime, LatestStateSender } from '../src/projects/matchRuntime.js';
import { TOURNAMENT_PROJECTS } from '../src/projects/catalog.js';
import { projectionIsOlder, receivedProjectView } from '../../shared/projectStateOrder.mjs';

const bootstrap = project => ({ instance_id: 'game:yellow', project_ref: project.id,
  rules_version: project.adapterRulesVersion || 'tournament-v2', seed: 'shared-match-seed', side: 'yellow', sequence: 0 });
const options = { now: () => 0, evilSpawn: async board => ({ index: board.indexOf(0), value: 2 }) };
const project = order => TOURNAMENT_PROJECTS.find(item => item.order === order);

for (const item of TOURNAMENT_PROJECTS) test(`${item.order}: local deterministic execution and checkpoint continuation`, async () => {
  const a = new MatchRuntime(bootstrap(item), options);
  const b = new MatchRuntime(bootstrap(item), options);
  assert.deepEqual(a.game.board, b.game.board);
  a.accept(); b.accept();
  for (const direction of ['left', 'down', 'right', 'up', 'left']) {
    const result = a.move(direction);
    if (!item.evilSpawn) assert.equal(Boolean(result?.then), false, 'normal input must execute synchronously');
    await result; await b.move(direction);
  }
  assert.deepEqual(a.checkpoint(), b.checkpoint());
  const resumed = new MatchRuntime({ ...bootstrap(item), sequence: a.sequence, checkpoint: a.checkpoint() }, options);
  for (const direction of ['up', 'right', 'down']) { await a.move(direction); await resumed.move(direction); }
  assert.deepEqual(a.checkpoint(), resumed.checkpoint());
  assert.deepEqual(a.packet().payload.board, resumed.packet().payload.board);
  assert.equal(a.packet().sequence, resumed.packet().sequence);
});

test('undo preserves RNG; restart and pause preserve match elapsed time', () => {
  let now = 0;
  const runtime = new MatchRuntime(bootstrap(project(5)), { now: () => now, elapsedMs: 1000 });
  runtime.move('left');
  const random = runtime.game.randomState;
  const sequence = runtime.sequence;
  runtime.action('undo');
  assert.equal(runtime.game.randomState, random);
  assert.equal(runtime.sequence, sequence + 1);
  now = 500;
  runtime.action('restart');
  assert.equal(runtime.elapsed(), 1500);
  runtime.setClock(1500, false);
  now = 1500;
  assert.equal(runtime.elapsed(), 1500);
  assert.equal(runtime.move('left'), null);
  runtime.setClock(1500, true);
  now = 1600;
  assert.equal(runtime.elapsed(), 1600);
});

test('restartable race death is not a terminal match result', () => {
  const runtime = new MatchRuntime(bootstrap(project(4)), options);
  runtime.game.finished = true; runtime.game.outcome = 'no_moves';
  assert.equal(runtime.accept().finished, false);
  assert.equal(runtime.action('restart').finished, false);
  runtime.game.finished = true; runtime.game.outcome = 'no_moves';
  assert.equal(runtime.stopRace().outcome, 'opponent_finished');
});

test('cargo countdown completes without a move and freezes at ten minutes', () => {
  let now = 0;
  const runtime = new MatchRuntime(bootstrap(project(1)), { now: () => now });
  now = 600010;
  const packet = runtime.tick();
  assert.equal(packet.finished, true);
  assert.equal(packet.elapsed_ms, 600000);
  assert.equal(packet.outcome, 'time_limit');
  now = 700000;
  assert.equal(runtime.elapsed(), 600000);
});

test('network transmission grace never grants extra local play time', () => {
  let now = 0;
  const runtime = new MatchRuntime({ ...bootstrap(project(5)), team_remaining_at_start_ms: 1000 }, { now: () => now });
  now = 1001;
  assert.equal(runtime.elapsed(), 1000);
  assert.equal(runtime.playable(), false);
  assert.equal(runtime.move('left'), null);
  assert.equal(runtime.action('restart'), null);
});

test('failed WASM computation restores board and RNG instead of substituting random spawn', async () => {
  const runtime = new MatchRuntime(bootstrap(project(3)), { ...options, evilSpawn: async () => { throw new Error('worker error'); } });
  const before = runtime.checkpoint();
  await assert.rejects(runtime.move('left'), /worker error/);
  assert.deepEqual(runtime.checkpoint(), before);
  runtime.game.evilSpawn = options.evilSpawn;
  assert.ok(await runtime.move('left'));
});

test('held upload does not prevent 50 local moves; queued snapshots coalesce and stay ordered', async () => {
  let release;
  const sent = [];
  const runtime = new MatchRuntime(bootstrap(project(5)), options);
  const sender = new LatestStateSender({ interval: 10000, send: packet => {
    sent.push(packet.sequence);
    return new Promise(resolve => { release = () => resolve({ accepted_sequence: packet.sequence }); });
  } });
  sender.push(runtime.accept());
  const first = sender.flush();
  for (let i = 0; i < 50; i++) sender.push(runtime.move(['left', 'down', 'right', 'up'][i % 4]) || runtime.action('restart'));
  assert.ok(runtime.sequence > 35);
  assert.deepEqual(sent, [1]);
  const latest = runtime.sequence;
  release(); await first;
  const second = sender.flush();
  assert.deepEqual(sent, [1, latest]);
  release(); await second; sender.close();
});

test('retry retains latest final state, not failed older state', async () => {
  let reject;
  const sent = [];
  const sender = new LatestStateSender({ interval: 10000, send: packet => {
    sent.push(packet.sequence);
    return sent.length === 1 ? new Promise((_resolve, fail) => { reject = fail; }) : Promise.resolve({});
  } });
  sender.push({ sequence: 1 }); const first = sender.flush();
  sender.push({ sequence: 4, finished: true });
  reject(new Error('offline')); await first;
  await sender.flush(); sender.close();
  assert.deepEqual(sent, [1, 4]);
});

test('receiver rejects older generations/sequences and skips incompatible transition animations', () => {
  const view = sequence => ({ generation: 2, sequence, payload: { last_transition: { kind: 'move' } } });
  const previous = view(10);
  assert.equal(receivedProjectView(previous, view(9)), previous);
  assert.equal(receivedProjectView(previous, { ...view(100), generation: 1 }), previous);
  assert.equal(receivedProjectView(previous, view(11)).payload.last_transition.kind, 'move');
  assert.equal(receivedProjectView(previous, view(15)).payload.last_transition.kind, 'restore');
  assert.equal(projectionIsOlder({ public_key: 'x', generation: 1, content_sequence: 8 },
    { public_key: 'x', generation: 1, content_sequence: 7 }), true);
  assert.equal(projectionIsOlder({ public_key: 'x', generation: 1, content_sequence: 8 },
    { public_key: 'x', generation: 2, content_sequence: 1 }), false);
});
