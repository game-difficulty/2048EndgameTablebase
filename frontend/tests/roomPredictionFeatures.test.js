import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {openPredictionMarket} from '../src/features/roomActivities/activityAvailability.js';

test('shared room hint handles both market protocols and hides unavailable activities',()=>{
  const ai={market:{id:'ai',status:'open',deadline:200}};
  const event={markets:[{id:'event',status:'open'}]};
  assert.equal(openPredictionMarket(ai,100,true,true).id,'ai');
  assert.equal(openPredictionMarket(event,100,true,true).id,'event');
  for(const data of [ai,event]) {
    assert.equal(openPredictionMarket(data,100,false,true),null);
    assert.equal(openPredictionMarket(data,100,true,false),null);
    assert.equal(openPredictionMarket({...data,available:false},100,true,true),null);
  }
  assert.equal(openPredictionMarket(ai,200,true,true),null);
  assert.equal(openPredictionMarket({markets:[{status:'settled'},{status:'void'},{status:'closed'}]},100,true,true),null);
});
test('room layer owns hint; both betting dialogs reuse history and expose opening',()=>{
  const read=p=>readFileSync(new URL('../src/features/roomActivities/'+p,import.meta.url),'utf8');
  assert.match(read('RoomActivities.vue'),/<PredictionDock/);
  for(const p of ['Predictions.vue','CompetitionPredictions.vue']) {
    assert.match(read(p),/<PredictionHistory :results="data.recent"/);
    assert.match(read(p),/defineExpose\(\{open,close\}\)/);
    assert.doesNotMatch(read(p),/<RoomActivityEntry/);
  }
});
