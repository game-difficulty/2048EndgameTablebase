import test from 'node:test';
import assert from 'node:assert/strict';
import { emptyHistoryFilters, historyFilterParams } from '../src/human/historyFilters.js';
test('history date filters include the last local calendar day', () => {
  const params = historyFilterParams({...emptyHistoryFilters(), minScore: '4000', from:'2026-10-01', to:'2026-10-02'});
  assert.equal(params.min_score, '4000');
  assert.equal(Number(params.time_from), new Date(2026,9,1).getTime()/1000);
  assert.equal(Number(params.time_to), new Date(2026,9,3).getTime()/1000);
  for (const extra of [{minScore:'-1'}, {minScore:'1.5'}, {from:'2026-02-30'}, {from:'2026-10-03',to:'2026-10-01'}]) assert.throws(()=>historyFilterParams({...emptyHistoryFilters(),...extra}));
  assert.equal(historyFilterParams(emptyHistoryFilters()).time_from, undefined);
});
