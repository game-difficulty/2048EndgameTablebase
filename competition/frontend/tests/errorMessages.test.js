import test from 'node:test';
import assert from 'node:assert/strict';

import { userFacingError } from '../src/errorMessages.js';

test('API and socket error codes have Chinese, actionable messages', () => {
  const apiError = Object.assign(new Error('Competition room not found.'), {
    code: 'ROOM_NOT_FOUND', status: 404,
  });
  assert.equal(userFacingError(apiError), '找不到该比赛房间，请检查房间码。');
  assert.equal(userFacingError({ code: 'SEAT_TAKEN', message: 'This seat is already occupied.' }), '该席位已有人落座。');
  assert.equal(userFacingError({ code: 'AUTH_REQUIRED', message: 'Authentication required.' }), '请先登录后再操作。');
});

test('unknown English diagnostics never appear directly in the UI', () => {
  assert.equal(userFacingError({ code: 'UNKNOWN_CODE', message: 'Unexpected backend detail.', status: 409 }), '房间状态已变化，请刷新后重试。');
  assert.equal(userFacingError(new TypeError('Failed to fetch')), '网络连接失败，请检查网络后重试。');
  assert.equal(userFacingError(new Error('Unexpected backend detail.')), '操作失败，请稍后重试。');
});

test('existing Chinese local validation messages remain intact', () => {
  assert.equal(userFacingError(new Error('项目池至少需要 5 个项目。')), '项目池至少需要 5 个项目。');
  assert.equal(userFacingError('房间连接失败'), '房间连接失败');
});
