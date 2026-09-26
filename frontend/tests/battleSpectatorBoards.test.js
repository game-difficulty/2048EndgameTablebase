import assert from 'node:assert/strict';
import test from 'node:test';
import { createStablePlayerOrder } from '../src/features/battle/core/stablePlayerOrder.js';
import { createObserverPlayback } from '../src/features/battle/core/observerPlayback.js';
import { observedFreeBoardView } from '../src/features/battle/modes/freeGoodness/observedTransition.js';
import { buildOptimisticMoveOnlyTransition, encodeBoard } from '../src/features/replay/engine/replayTransition.js';

test('spectator seats stay fixed while live goodness order changes', () => {
  const order = createStablePlayerOrder();
  const room = { room_id: 'room', round: { round_id: 'first' },
    members: [{ actor_key: 'u:2', seat_index: 1 }, { actor_key: 'u:1', seat_index: 0 }],
    results: [{ actor_key: 'u:2', goodness_of_fit: 1 }, { actor_key: 'u:1', goodness_of_fit: .8 }] };
  assert.deepEqual(order(room).map(row => row.actor_key), ['u:1', 'u:2']);
  room.results.reverse();
  room.results[0].goodness_of_fit = .95;
  assert.deepEqual(order(room).map(row => row.actor_key), ['u:1', 'u:2']);
  room.members = [{ actor_key: 'u:2', seat_index: 1 }];
  assert.deepEqual(order(room).map(row => row.actor_key), ['u:1', 'u:2']);
  room.round.round_id = 'second';
  assert.deepEqual(order(room).map(row => row.actor_key), ['u:2', 'u:1']);
});

test('free-mode spectator uses slide, merge, and spawn metadata for one new step', () => {
  const before = [2, 2, ...Array(14).fill(0)];
  const moved = buildOptimisticMoveOnlyTransition(before, 'left', false);
  const after = moved.board.slice();
  after[1] = 2;
  const hex = board => encodeBoard(board).toString(16).padStart(16, '0');
  const result = { actor_key: 'u:1', route_index: 0, last_sequence: 0,
    mode_data: { board_hex: hex(before) } };
  const room = { round: { round_id: 'round' }, results: [result] };
  let frames;
  const observer = createObserverPlayback({
    resolve: (player, now) => observedFreeBoardView(player, now, false),
    canSee: () => true, publish: value => { frames = value; },
  });
  observer.update(room);
  result.route_index = 1;
  result.last_sequence = 1;
  result.mode_data = { board_hex: hex(after), last_step: {
    sequence: 1, previous_board_hex: hex(before), board_hex: hex(after),
    executed_direction: 'left', spawn_index: 1, spawn_value: 2,
  } };
  observer.update(room);
  assert.equal(frames['u:1'].kind, 'move');
  assert.equal(frames['u:1'].metadata.direction, 'left');
  assert.ok(frames['u:1'].metadata.slide_distances.some(distance => distance > 0));
  assert.ok(frames['u:1'].metadata.pop_positions.some(position => position > 0));
  assert.deepEqual(frames['u:1'].metadata.appear_tile, { index: 1, value: 2 });
  const animated = frames['u:1'];
  observer.update(room);
  assert.equal(frames['u:1'], animated);
  observer.clear();
});

test('free-mode correction animates after dismissal but a missed step snaps safely', () => {
  const before = [2, 2, ...Array(14).fill(0)];
  const moved = buildOptimisticMoveOnlyTransition(before, 'left', false);
  const after = moved.board.slice();
  after[1] = 2;
  const hex = board => encodeBoard(board).toString(16).padStart(16, '0');
  const result = { actor_key: 'u:1', route_index: 1, last_sequence: 1, status: 'playing',
    mode_data: { board_hex: hex(after), last_step: {
      sequence: 1, previous_board_hex: hex(before), board_hex: hex(after),
      executed_direction: 'left', spawn_index: 1, spawn_value: 2,
    }, correction: { selected_direction: 'right', standard_direction: 'left',
      previous_board_hex: hex(before), visible_until: new Date(Date.now() + 15_000).toISOString() } } };
  const room = { round: { round_id: 'round' }, results: [result] };
  let frames;
  const observer = createObserverPlayback({
    resolve: (player, now) => observedFreeBoardView(player, now, false),
    canSee: () => true, publish: value => { frames = value; },
  });
  observer.update(room);
  assert.equal(frames['u:1'].kind, 'snapshot');
  delete result.mode_data.correction;
  observer.update(room);
  assert.equal(frames['u:1'].kind, 'move');
  result.route_index = 3;
  observer.update(room);
  assert.equal(frames['u:1'].kind, 'snapshot');
  observer.clear();
});
