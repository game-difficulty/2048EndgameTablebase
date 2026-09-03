import assert from 'node:assert/strict';
import test from 'node:test';

import {
  buildTrainerQueryPayloadForActor,
  createTrainerLookupAccessCoordinator,
  isTrainerPatternGuestAvailable,
  isTrainerTableGuestAvailable,
  normalizeTrainerGuestAllowance,
  trainerGuestErrorNotice,
} from '../src/features/trainer/engine/trainerGuestDemo.js';

const tables = [
  { pattern: 'L3', target: '128', fullPattern: 'L3_128', guestAvailable: true },
  { pattern: 'L3', target: '256', fullPattern: 'L3_256', guestAvailable: false },
  { pattern: 'free10', target: '128', full_pattern: 'free10_128', guest_available: false },
];

test('guest table availability is strict while keeping mixed patterns discoverable', () => {
  assert.equal(isTrainerTableGuestAvailable(tables, 'L3_128'), true);
  assert.equal(isTrainerTableGuestAvailable(tables, 'L3_256'), false);
  assert.equal(isTrainerTableGuestAvailable(tables, 'missing_128'), false);
  assert.equal(isTrainerPatternGuestAvailable(tables, 'L3'), true);
  assert.equal(isTrainerPatternGuestAvailable(tables, 'free10'), false);
});

test('guest foreground queries explicitly disable prefetch without changing user payloads', () => {
  const payload = {
    board_hex: '0000000000000011',
    prefetch_rng: { state: [1, 2, 3, 4] },
  };
  assert.deepEqual(buildTrainerQueryPayloadForActor(payload, { kind: 'guest' }), {
    board_hex: '0000000000000011',
    prefetch_rng: null,
    allow_prefetch: false,
    guest_foreground_query: true,
  });
  assert.equal(buildTrainerQueryPayloadForActor(payload, { kind: 'user' }), payload);
});

test('guest allowance is normalized and error codes map to stable notices', () => {
  assert.deepEqual(normalizeTrainerGuestAllowance({ remaining: 3.9, total: 5 }), {
    remaining: 3,
    total: 5,
  });
  assert.deepEqual(normalizeTrainerGuestAllowance({ remaining: 4, limit: 5, used: 1 }), {
    remaining: 4,
    total: 5,
  });
  assert.equal(normalizeTrainerGuestAllowance({ remaining: 'bad', total: 5 }, null), null);
  assert.equal(trainerGuestErrorNotice('GUEST_QUERY_LIMIT_REACHED'), 'exhausted');
  assert.equal(trainerGuestErrorNotice('GUEST_QUERY_ALLOWANCE_EXHAUSTED'), 'exhausted');
  assert.equal(trainerGuestErrorNotice('GUEST_NETWORK_QUERY_LIMIT_REACHED'), 'exhausted');
  assert.equal(trainerGuestErrorNotice('GUEST_TABLE_NOT_AVAILABLE'), 'locked');
  assert.equal(trainerGuestErrorNotice('GUEST_TABLE_LOGIN_REQUIRED'), 'locked');
  assert.equal(trainerGuestErrorNotice('OTHER'), '');
});

test('first protected lookup creates one guest session and replays only the latest keyed action', async () => {
  let actor = null;
  let ensureCalls = 0;
  const sent = [];
  const coordinator = createTrainerLookupAccessCoordinator({
    getActor: () => actor,
    ensureGuestSession: async () => {
      ensureCalls += 1;
      actor = { kind: 'guest' };
      return actor;
    },
    validateActor: () => true,
  });

  assert.equal(coordinator.request('query', () => sent.push('stale')), true);
  assert.equal(coordinator.request('query', () => sent.push('latest')), true);
  await new Promise((resolve) => setTimeout(resolve, 0));

  assert.equal(ensureCalls, 1);
  assert.deepEqual(sent, ['latest']);
});

test('registered actors execute immediately without creating guest sessions', () => {
  let ensureCalls = 0;
  let sentActor = null;
  const user = { kind: 'user', user_id: 7 };
  const coordinator = createTrainerLookupAccessCoordinator({
    getActor: () => user,
    ensureGuestSession: async () => {
      ensureCalls += 1;
      return { kind: 'guest' };
    },
    validateActor: () => true,
  });

  assert.equal(coordinator.request('query', (actor) => { sentActor = actor; }), true);
  assert.equal(sentActor, user);
  assert.equal(ensureCalls, 0);
});
