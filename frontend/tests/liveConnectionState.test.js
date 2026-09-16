import test from 'node:test';
import assert from 'node:assert/strict';
import { liveConnectionState } from '../src/live/connectionState.js';

test('administrative pause is distinct from disconnection and waits for a snapshot', () => {
  assert.equal(liveConnectionState({ paused:true }), 'loading');
  assert.equal(liveConnectionState({ connected:true,synchronized:true,paused:true }), 'paused');
  assert.equal(liveConnectionState({ connected:true,synchronized:true,paused:false,online:true }), 'live');
});

test('initial state and socket open without a snapshot never imply broadcaster offline', () => {
  assert.equal(liveConnectionState({}), 'loading');
  assert.equal(liveConnectionState({ connected:true, online:false }), 'loading');
});
test('only a synchronized connection can confirm live or offline', () => {
  const state={connected:true,synchronized:true,seenSnapshot:true};
  assert.equal(liveConnectionState({...state,online:true}), 'live');
  assert.equal(liveConnectionState({...state,online:false}), 'offline');
});
test('lost connection and pending reconnect snapshot use reconnecting regardless of prior broadcaster status', () => {
  for(const online of [false,true]) {
    assert.equal(liveConnectionState({seenSnapshot:true,online}), 'reconnecting');
    assert.equal(liveConnectionState({connected:true,seenSnapshot:true,online,synchronized:false}), 'reconnecting');
  }
});
