import test from 'node:test';
import assert from 'node:assert/strict';
import { recoverAnalysisJob } from '../src/human/analysisJobRecovery.js';

test('expired remembered task does not block selecting new analyses or reading saved summaries', async () => {
  assert.equal(await recoverAnalysisJob(async () => { throw { status: 404 }; }, 'expired'), null);
});

test('existing queued, running and completed tasks can still be resumed', async () => {
  for (const status of ['queued', 'running', 'finished']) {
    const job = { job_id: 'remembered', status };
    assert.equal(await recoverAnalysisJob(async id => {
      assert.equal(id, job.job_id);
      return job;
    }, job.job_id), job);
  }
});

test('authentication, server and network failures remain actionable errors', async () => {
  for (const error of [{ status: 401 }, { status: 403 }, { status: 500 }, new TypeError('Failed to fetch')]) {
    await assert.rejects(recoverAnalysisJob(async () => { throw error; }, 'remembered'), actual => actual === error);
  }
});
