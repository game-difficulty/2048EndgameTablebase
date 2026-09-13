import assert from 'node:assert/strict';
import test from 'node:test';
import { createObserverPlayback } from '../src/features/battle/core/observerPlayback.js';
import { createRoutePresentation } from '../src/features/battle/modes/goodness/engine/routePresentation.js';
import { BattleController } from '../src/features/battle/modes/goodness/engine/battleController.js';
import { encodeBoard } from '../src/features/replay/engine/replayTransition.js';
import { spectatorLayout } from '../src/features/battle/core/spectatorLayout.js';

function fixture() {
  const route = { format: 'trainer-route-v1', initialBoard: encodeBoard([2, 2, ...Array(14).fill(0)]),
    moveCount: 3, changes: new Uint8Array([2 | (15 << 2), (15 << 2), 2 | (15 << 2)]),
    rates: new Uint32Array(12).fill(2_000_000_000) };
  return new BattleController({ route, certaintyStep: 1 });
}
function setup({ completed = false } = {}) {
  let time = 1000;
  let timer;
  let visible;
  const controller = fixture();
  const result = { actor_key: 'u:1', route_index: completed ? 3 : 1,
    status: completed ? 'completed' : 'playing', mode_data: {
      correction: { selected_direction: 'right', standard_direction: 'left',
        previous_route_index: 0, visible_until: new Date(16000).toISOString() },
      ...(completed ? { auto_playback: { from_index: 1, started_at: new Date(16000).toISOString(), step_ms: 150 } } : {}),
    } };
  const room = { round: { round_id: 'round-a' }, results: [result] };
  const observer = createObserverPlayback({ resolve: createRoutePresentation(controller),
    canSee: () => true, publish: (frames, overlays) => { visible = { frames, overlays }; },
    now: () => time, schedule: (fn, ms) => { timer = { fn, at: time + ms }; return 1; },
    cancel: () => { timer = null; } });
  observer.update(room);
  return { observer, room, result, controller,
    view: () => visible, timer: () => timer,
    tick(at) { time = at; const task = timer; timer = null; task?.fn(); },
  };
}

test('early dismissal updates board at same server index and animates only once', () => {
  const f = setup();
  const before = f.view().frames['u:1'];
  delete f.result.mode_data.correction;
  f.observer.update(f.room);
  const after = f.view().frames['u:1'];
  assert.notEqual(after.revision, before.revision);
  assert.notDeepEqual(after.toBoard, before.toBoard);
  assert.equal(after.kind, 'move');
  assert.deepEqual(f.view().overlays, {});
  f.observer.update(f.room);
  assert.equal(f.view().frames['u:1'], after);
  assert.equal(f.timer(), null);
});

test('expiry removes overlay and advances board without another room message', () => {
  const f = setup();
  f.tick(16000);
  assert.deepEqual(f.view().overlays, {});
  assert.equal(f.view().frames['u:1'].kind, 'move');
  assert.deepEqual(f.view().frames['u:1'].toBoard, f.controller.seek(1).board);
  assert.equal(f.timer(), null);
});

test('completed correction plays automatic tail locally then stops scheduling', () => {
  const f = setup({ completed: true });
  f.tick(16000);
  assert.deepEqual(f.view().frames['u:1'].toBoard, f.controller.seek(1).board);
  f.tick(16150);
  assert.deepEqual(f.view().frames['u:1'].toBoard, f.controller.seek(2).board);
  f.tick(16300);
  assert.deepEqual(f.view().frames['u:1'].toBoard, f.controller.seek(3).board);
  assert.equal(f.timer(), null);
});

test('joining late skips to current tail position, and a new round cannot replay old animation', () => {
  const f = setup({ completed: true });
  delete f.result.mode_data.correction;
  f.result.mode_data.auto_playback.started_at = new Date(0).toISOString();
  f.room.round.round_id = 'round-b';
  f.observer.update(f.room);
  assert.equal(f.view().frames['u:1'].kind, 'snapshot');
  assert.deepEqual(f.view().frames['u:1'].toBoard, f.controller.seek(3).board);
  f.observer.clear();
  assert.deepEqual(f.view(), { frames: {}, overlays: {} });
  assert.equal(f.timer(), null);
});

test('a delayed correction-close timestamp cannot move the displayed automatic tail backwards', () => {
  const f = setup({ completed: true });
  f.tick(16150);
  delete f.result.mode_data.correction;
  f.result.mode_data.auto_playback.started_at = new Date(16000).toISOString();
  f.observer.update(f.room);
  assert.deepEqual(f.view().frames['u:1'].toBoard, f.controller.seek(2).board);
});

test('spectator layout fills seats symmetrically for one through eight players', () => {
  for (let count = 1; count <= 8; count += 1) {
    assert.equal(spectatorLayout(count).items.length, count);
  }
  assert.equal(spectatorLayout(2).items[0].gridColumn, 'span 12');
  assert.equal(spectatorLayout(3).items[0].gridColumn, 'span 8');
  assert.equal(spectatorLayout(5).items[3].gridColumn, '5 / span 8');
  assert.equal(spectatorLayout(7).items[4].gridColumn, '4 / span 6');
});
