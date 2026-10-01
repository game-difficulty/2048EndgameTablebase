import test from 'node:test';
import assert from 'node:assert/strict';
import { MatchRuntime } from '../src/projects/matchRuntime.js';
import { ALL_PROJECTS as TOURNAMENT_PROJECTS } from '../src/projects/catalog.js';
import { projectionIsOlder, receivedProjectView } from '../../shared/projectStateOrder.mjs';

const bootstrap = project => ({ instance_id: 'game:yellow', project_ref: project.id,
  rules_version: project.adapterRulesVersion || 'tournament-v2', seed: 'shared-match-seed', side: 'yellow', sequence: 0 });
const options = { now: () => 0, evilSpawn: async board => ({ index: board.indexOf(0), value: 2 }) };
const project = order => TOURNAMENT_PROJECTS.find(item => item.order === order);

test('Higher time is never extended in play; checkpoint timelines remain immutable', () => {
  let time=0;
  const runtime=new MatchRuntime({...bootstrap(project(1)),team_remaining_at_start_ms:60000},{now:()=>time});
  const first=runtime.accept();
  time=1500;runtime.game.score=3;runtime.accept();
  time=59999;
  assert.equal(runtime.playable(),true);
  time=60000;
  assert.equal(runtime.playable(),false);
  assert.equal(runtime.move('down'),null);
  assert.equal(runtime.action('surrender'),null);
  time=70000;
  assert.equal(runtime.elapsed(),60000);
  assert.equal(runtime.budget(),60000);
  assert.deepEqual(first.checkpoint.metric_history,[[0,0]]);
  const resumed=new MatchRuntime({...bootstrap(project(1)),team_remaining_at_start_ms:60000,checkpoint:runtime.checkpoint()}, {elapsedMs:70000,now:()=>time});
  assert.equal(resumed.playable(),false);
  assert.equal(resumed.elapsed(),60000);
});

test('surrender freezes time and preserves the current score and board', () => {
  let time = 0;
  const runtime = new MatchRuntime(bootstrap(project(2)), { now: () => time });
  runtime.game.score = 4321;
  const board = [...runtime.game.board];
  time = 1234;
  const packet = runtime.action('surrender');
  time = 9876;
  assert.equal(packet.outcome, 'surrendered');
  assert.equal(packet.result_value, 4321);
  assert.equal(packet.finished, true);
  assert.deepEqual(packet.payload.board.flat(), board);
  assert.equal(runtime.elapsed(), 1234);
  assert.equal(runtime.move('left'), null);
  assert.equal(runtime.action('restart'), null);
});

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

test('chemical color cursor survives a checkpoint and continues the same future sequence', () => {
  const item = project(14);
  const original = new MatchRuntime(bootstrap(item), options);
  original.game.project = { ...original.game.project, specialSpawnRate: 1 };
  for (let i = 0; i < 7; i++) { original.game.tiles = []; original.game.spawn(); }
  const checkpoint = original.checkpoint();
  assert.equal(checkpoint.state.chemicalColorState, original.game.chemicalColorState);
  const resumed = new MatchRuntime({ ...bootstrap(item), checkpoint }, options);
  resumed.game.project = { ...resumed.game.project, specialSpawnRate: 1 };
  for (let i = 0; i < 20; i++) {
    original.game.tiles = []; resumed.game.tiles = [];
    assert.deepEqual(resumed.game.spawn(), original.game.spawn());
  }
});

test('fission timers survive a match checkpoint just before splitting', () => {
  const item = project(18);
  const original = new MatchRuntime(bootstrap(item), options);
  original.game.board = [1024,0,0,0, 0,2,0,0, ...Array(8).fill(0)];
  original.game.fissionTimers = new Map([[0, { sequence: 0, remaining: 1 }]]);
  original.game.fissionSequence = 1;
  const resumed = new MatchRuntime({ ...bootstrap(item), checkpoint: original.checkpoint() }, options);
  assert.deepEqual(resumed.checkpoint(), original.checkpoint());
  assert.deepEqual(resumed.move('left')?.payload.board, original.move('left')?.payload.board);
  assert.deepEqual(resumed.game.transition.fission, original.game.transition.fission);
});

test('aftershock coordinates and its independent draw state survive a match checkpoint', () => {
  const item = project(19);
  const original = new MatchRuntime(bootstrap(item), options);
  original.game.board = [128,128,0,0, 2,0,0,0, ...Array(8).fill(0)];
  const moved = original.move('left');
  assert.ok(moved.payload.last_transition.quake);
  assert.equal(moved.payload.shape_shifter, true);
  assert.equal(moved.payload.last_transition.kind, 'reshape');
  const resumed = new MatchRuntime({ ...bootstrap(item), checkpoint: moved.checkpoint, sequence: moved.sequence }, options);
  assert.deepEqual(resumed.checkpoint().state.board, moved.checkpoint.state.board);
  assert.equal(resumed.game.originCol, original.game.originCol);
  assert.equal(resumed.game.quakeState, original.game.quakeState);
  assert.deepEqual(resumed.game.move('down').snapshot.board, original.game.move('down').snapshot.board);
});

test('look-back history and chance state survive a match checkpoint', () => {
  const item = project(20);
  const original = new MatchRuntime(bootstrap(item), options);
  original.game.board = [2,0,0,0, ...Array(8).fill(0)];
  const moved = original.move('right');
  assert.equal(moved.checkpoint.state.lookBackHistory.length, 1);
  const resumed = new MatchRuntime({ ...bootstrap(item), checkpoint: moved.checkpoint, sequence: moved.sequence }, options);
  assert.deepEqual(resumed.checkpoint().state.lookBackHistory, moved.checkpoint.state.lookBackHistory);
  assert.equal(resumed.game.lookBackState, original.game.lookBackState);
  assert.deepEqual(resumed.game.move('left').snapshot.board, original.game.move('left').snapshot.board);
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

test('cargo has no project deadline and continues beyond ten minutes', () => {
  let now = 0;
  const runtime = new MatchRuntime(bootstrap(project(1)), { now: () => now });
  now = 600010;
  const packet = runtime.tick();
  assert.equal(packet, null);
  assert.equal(runtime.completed(),false);
  now = 700000;
  assert.equal(runtime.elapsed(), 700000);
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
