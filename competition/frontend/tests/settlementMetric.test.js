import test from 'node:test';
import assert from 'node:assert/strict';
import { settlementMetric } from '../../shared/settlementMetric.mjs';
test('settlement distinguishes score, cargo, board sum and race timing',()=>{
  assert.equal(settlementMetric({yellow_score:12345},'yellow').value,'12,345');
  assert.equal(settlementMetric({yellow_score:8},'yellow',{project_ref:'tournament-cargo-transport-4x4'}).label,'送出数量');
  assert.equal(settlementMetric({yellow_score:1022},'yellow',{project_ref:'tournament-dice-wall-3x4'}).label,'盘面和');
  assert.equal(settlementMetric({reason:'race_elapsed',yellow_outcome:'target_reached',yellow_elapsed_ms:114514},'yellow').value,'01:54.51');
});
test('DNF, missing times, forfeits and corrections do not look like zero-second wins',()=>{
  const project={project_ref:'tournament-pure2-full-race-3x3'};
  assert.equal(settlementMetric({yellow_score:0,reason:'white_clock_expired'},'yellow',project).value,'DNF');
  assert.equal(settlementMetric({yellow_outcome:'target_reached'},'yellow',project).value,'—');
  assert.equal(settlementMetric({reason:'late_forfeit'},'yellow',project).note,'未开赛');
  assert.equal(settlementMetric({reason:'yellow_surrendered',yellow_score:28},'yellow').value,'28');
  assert.equal(settlementMetric({corrected:true,yellow_score:100},'yellow',project).label,'裁定成绩');
});

test('exhausted teams show frozen Higher metrics or Faster DNF on both sites',()=>{
  const result={yellow_outcome:'time_limit',yellow_score:1234};
  assert.deepEqual(settlementMetric(result,'yellow'),{label:'得分',value:'1,234',note:'包干时间耗尽'});
  const race=settlementMetric({...result,reason:'race_unfinished'},'yellow',{},'en');
  assert.equal(race.value,'DNF');
  assert.equal(race.note,'TEAM TIME EXPIRED');
});
