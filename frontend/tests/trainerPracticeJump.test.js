import assert from 'node:assert/strict';
import test from 'node:test';

import {
  clearTrainerPracticeContext,
  queueTrainerPracticeJump,
  registerTrainerPracticeContextConsumer,
  registerTrainerPracticeJumpConsumer,
  resetTrainerPracticeJumpQueue,
} from '../src/features/trainer/services/trainerPracticeJump.js';

test.afterEach(() => {
  resetTrainerPracticeJumpQueue();
});

test('publishes and clears the Battle practice context independently of the jump', () => {
  const received = [];
  queueTrainerPracticeJump({
    fullPattern: 'L3_128',
    hex: '0000000000000011',
    context: { kind: 'battle', roomId: 'room-1', fullPattern: 'L3_128' },
  });
  registerTrainerPracticeContextConsumer((context) => received.push(context));
  clearTrainerPracticeContext('battle');
  assert.deepEqual(received, [
    { kind: 'battle', roomId: 'room-1', fullPattern: 'L3_128' },
    null,
  ]);
});

test('delivers a queued jump when Trainer mounts later', () => {
  const received = [];
  queueTrainerPracticeJump({
    fullPattern: 'free10-512',
    hex: '001064127ff5ffff',
  });
  registerTrainerPracticeJumpConsumer((detail) => received.push(detail));
  assert.deepEqual(received, [{
    fullPattern: 'free10-512',
    hex: '001064127ff5ffff',
  }]);
});

test('delivers jumps immediately while Trainer is mounted', () => {
  const received = [];
  const unregister = registerTrainerPracticeJumpConsumer((detail) => received.push(detail));
  queueTrainerPracticeJump({ hex: '100021103456a987' });
  unregister();
  assert.deepEqual(received, [{ hex: '100021103456a987' }]);
});

test('keeps only the newest jump until Trainer can consume it', () => {
  const received = [];
  queueTrainerPracticeJump({ hex: '1111111111111111' });
  queueTrainerPracticeJump({ hex: '2222222222222222' });
  registerTrainerPracticeJumpConsumer((detail) => received.push(detail));
  assert.deepEqual(received, [{ hex: '2222222222222222' }]);
});
