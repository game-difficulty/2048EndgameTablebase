import test from 'node:test';
import assert from 'node:assert/strict';
import { predictionIsOpen, envelopeIsClaimable } from '../src/features/roomActivities/activityAvailability.js';

test('prediction dock exists only during the connected, live entry window', () => {
  const market={status:'open',deadline:100};
  assert.equal(predictionIsOpen(market,99,true,true),true);
  for(const status of ['closed','void','settled']) assert.equal(predictionIsOpen({...market,status},99,true,true),false);
  assert.equal(predictionIsOpen(market,100,true,true),false);
  assert.equal(predictionIsOpen(market,99,false,true),false);
  assert.equal(predictionIsOpen(market,99,true,false),false);
  assert.equal(predictionIsOpen(null,99,true,true),false);
});

test('red envelope dock only advertises claimable shares (also before next server tick)', () => {
  const envelope={status:'active',expires_at:100,sender_id:2,count:5,claimed:0};
  assert.equal(envelopeIsClaimable(envelope,99,true,1),true);
  assert.equal(envelopeIsClaimable(envelope,99,true,undefined),true);
  for(const status of ['queued','exhausted','expired','closed']) assert.equal(envelopeIsClaimable({...envelope,status},99,true,1),false);
  assert.equal(envelopeIsClaimable(envelope,100,true,1),false);
  assert.equal(envelopeIsClaimable({...envelope,claimed:5},99,true,1),false);
  assert.equal(envelopeIsClaimable(envelope,99,false,1),false);
  assert.equal(envelopeIsClaimable(envelope,99,true,2),false);
  assert.equal(envelopeIsClaimable(envelope,99,true,1,true),false);
  assert.equal(envelopeIsClaimable(null,99,true,1),false);
});
