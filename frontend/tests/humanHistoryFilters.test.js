import test from 'node:test';
import assert from 'node:assert/strict';
import { emptyHistoryFilters, historyFilterParams, completeHistoryDates, historyPageTarget } from '../src/human/historyFilters.js';
test('history date filters include the last local calendar day', () => {
  const params = historyFilterParams({...emptyHistoryFilters(), minScore: '4000', from:'2026-10-01', to:'2026-10-02'});
  assert.equal(params.min_score, '4000');
  assert.equal(Number(params.time_from), new Date(2026,9,1).getTime()/1000);
  assert.equal(Number(params.time_to), new Date(2026,9,3).getTime()/1000);
  for (const extra of [{minScore:'-1'}, {minScore:'1.5'}, {from:'2026-02-30'}, {from:'2026-10-03',to:'2026-10-01'}]) assert.throws(()=>historyFilterParams({...emptyHistoryFilters(),...extra}));
  assert.equal(historyFilterParams(emptyHistoryFilters()).time_from, undefined);
});

test('date completion copies only into an empty opposite field', () => {
  for (const key of ['from','to']) {
    const f=emptyHistoryFilters();f[key]='2026-10-03';completeHistoryDates(f,key);
    assert.equal(f.from,'2026-10-03');assert.equal(f.to,'2026-10-03');
    f.from='2026-10-01';completeHistoryDates(f,'to');assert.equal(f.from,'2026-10-01');
  }
  const f=emptyHistoryFilters();completeHistoryDates(f,'from');assert.equal(f.to,'');
});
test('page jumps clamp bounds and reject empty or fractional values', () => {
  assert.equal(historyPageTarget(-9,25),1);
  assert.equal(historyPageTarget(35,25),25);
  assert.equal(historyPageTarget('12',25),12);
  for (const value of ['', ' ', 'abc', '2.5', Infinity]) assert.equal(historyPageTarget(value,25),null);
});
