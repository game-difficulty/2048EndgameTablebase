import test from 'node:test';
import assert from 'node:assert/strict';
import { isRoomEndedEvent } from '../src/live/roomLifecycle.js';

test('only a matching explicit room-ended event ends the current room', () => {
  assert.equal(isRoomEndedEvent({ type: 'room_ended', room_id: 'h-one' }, 'h-one'), true);
  assert.equal(isRoomEndedEvent({ type: 'room_ended' }, 'h-one'), true);
  assert.equal(isRoomEndedEvent({ type: 'room_ended', room_id: 'h-two' }, 'h-one'), false);
  assert.equal(isRoomEndedEvent({ type: 'presence', online: false }, 'h-one'), false);
  assert.equal(isRoomEndedEvent(null, 'h-one'), false);
});
