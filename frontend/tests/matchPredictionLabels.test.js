import {test} from 'node:test';
import assert from 'node:assert/strict';
import {marketTitle} from '../src/features/roomActivities/matchPredictionLabels.js';
test('BO5 and BO7 titles specify first threshold rather than final score',()=>{
  assert.match(marketTitle('clinch_3','zh'),/首次达到 3 胜/);
  assert.match(marketTitle('clinch_4','en'),/first 4 wins/);
  for(const kind of ['winner','first_two','clinch_3','clinch_4'])for(const lang of ['zh','en'])assert.notEqual(marketTitle(kind,lang),kind);
});
