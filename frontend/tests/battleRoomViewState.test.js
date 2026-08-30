import assert from 'node:assert/strict';
import test from 'node:test';

import { createBattleRoomViewState } from '../src/features/battle/core/battleRoomViewState.js';

function room({ roomId = 'room-1', roomStatus, roundId = 'round-1', roundStatus }) {
  return {
    room_id: roomId,
    status: roomStatus,
    round: { round_id: roundId, status: roundStatus },
  };
}

test('a participant stays on the completed match until returning to the lobby', () => {
  const state = createBattleRoomViewState();
  const running = room({ roomStatus: 'running', roundStatus: 'running' });
  const completed = room({ roomStatus: 'waiting', roundStatus: 'completed' });

  assert.deepEqual(state.apply(null, running), { heldRoundId: '', resultRoundId: '' });
  assert.deepEqual(state.apply(running, completed), {
    heldRoundId: 'round-1',
    resultRoundId: '',
  });
  assert.deepEqual(state.openResults(completed), {
    heldRoundId: 'round-1',
    resultRoundId: 'round-1',
  });
  assert.deepEqual(state.closeResults(), {
    heldRoundId: 'round-1',
    resultRoundId: '',
  });
  assert.deepEqual(state.returnToLobby(completed), {
    heldRoundId: '',
    resultRoundId: '',
  });
  assert.deepEqual(state.apply(completed, completed), {
    heldRoundId: '',
    resultRoundId: '',
  });
});

test('loading an already completed room opens the lobby instead of an old match', () => {
  const state = createBattleRoomViewState();
  const completed = room({ roomStatus: 'waiting', roundStatus: 'completed' });
  assert.deepEqual(state.apply(null, completed), {
    heldRoundId: '',
    resultRoundId: '',
  });
});

test('a new running round clears the previous completed view choice', () => {
  const state = createBattleRoomViewState();
  const firstRunning = room({ roomStatus: 'running', roundStatus: 'running' });
  const firstCompleted = room({ roomStatus: 'waiting', roundStatus: 'completed' });
  const nextRunning = room({
    roomStatus: 'running',
    roundId: 'round-2',
    roundStatus: 'running',
  });
  state.apply(null, firstRunning);
  state.apply(firstRunning, firstCompleted);
  state.openResults(firstCompleted);
  assert.deepEqual(state.apply(firstCompleted, nextRunning), {
    heldRoundId: '',
    resultRoundId: '',
  });
});
