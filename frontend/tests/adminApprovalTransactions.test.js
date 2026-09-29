import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const page = readFileSync(new URL('../src/features/admin/pages/AdminPage.vue', import.meta.url), 'utf8');
const panel = readFileSync(new URL('../src/features/admin/components/AdminApprovalTransactions.vue', import.meta.url), 'utf8');
const client = readFileSync(new URL('../src/services/admin/adminClient.js', import.meta.url), 'utf8');

test('admin page switches user lookup and approval transactions in one workspace', () => {
  assert.match(page, /adminSection === 'users'/);
  assert.match(page, /adminSection === 'approvals'/);
  assert.match(page, /<AdminApprovalTransactions/);
  assert.match(page, /v-if="adminSection === 'users'"/);
  assert.match(page, /<AdminApprovalTransactions[\s\S]*?v-else-if="adminSection === 'approvals'"/);
});

test('approval transaction panel queries summaries and reuses audited decision endpoints', () => {
  assert.match(client, /\/api\/admin\/approval-transactions/);
  assert.match(panel, /decideVerseClaim/);
  assert.match(panel, /retryVerseClaim/);
  assert.match(panel, /revokeVerseClaim/);
  assert.match(panel, /decideArchiveApplication/);
  assert.match(panel, /revokeArchiveApplication/);
  assert.doesNotMatch(panel, /replay|archive BLOB/i);
});
