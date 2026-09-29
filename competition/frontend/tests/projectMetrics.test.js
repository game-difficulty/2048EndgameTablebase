import test from 'node:test';
import assert from 'node:assert/strict';
import { projectPerformanceMetric, projectResultValue } from '../../shared/projectMetrics.mjs';

test('the primary metric follows each project objective', () => {
  assert.deepEqual(projectPerformanceMetric({ view_protocol: 'cargo-transport-v1', payload: { deliveries: 3, score: 90 } }), { label: '已送出', value: '3' });
  assert.deepEqual(projectPerformanceMetric({ payload: { result_metric: 'board_sum', board_sum: 2044, score: 400 } }), { label: '盘面和', value: '2,044' });
  assert.deepEqual(projectPerformanceMetric({ payload: { target_sum: 1022, board_sum: 1024, score: 400 } }), { label: '盘面和', value: '1,024' });
  assert.deepEqual(projectPerformanceMetric({ payload: { target_tile: 64, target_count: 10, current_target_count: 7, score: 5000 } }), { label: '64砖数量', value: '7' });
  assert.deepEqual(projectPerformanceMetric({ payload: { score: 3180 } }), { label: '得分', value: '3,180' });
});

test('race results show local completion time; referee corrections still show corrected scores', () => {
  const result = { reason: 'race_elapsed', yellow_elapsed_ms: 28403, white_elapsed_ms: 30000,
    yellow_outcome: 'target_reached', white_outcome: 'opponent_finished', yellow_score: 500 };
  assert.equal(projectResultValue(result, 'yellow'), '00:28.40');
  assert.equal(projectResultValue(result, 'white'), '未达标');
  assert.equal(projectResultValue({ ...result, corrected: true }, 'yellow'), '500');
});
