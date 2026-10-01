import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse, compileScript, compileTemplate } from '@vue/compiler-sfc';

const filename = new URL('../src/RoomMemberManagement.vue', import.meta.url);
const source = readFileSync(filename, 'utf8');

test('member management component compiles with accessible grouped controls', () => {
  const { descriptor, errors } = parse(source);
  assert.deepEqual(errors, []);
  const script = compileScript(descriptor, { id: 'members' });
  const result = compileTemplate({ source: descriptor.template.content, filename: filename.pathname, id: 'members', compilerOptions: { bindingMetadata: script.bindings } });
  assert.deepEqual(result.errors, []);
  assert.match(source, /<label>[\s\S]*<input/);
  assert.match(source, /:aria-label=/);
});

test('member controls retain self-removal and busy protections and emit the correct IDs', () => {
  assert.match(source, /busy \|\| seat\.user_id === currentUserId/);
  assert.match(source, /Number\(userId\.value\) !== props\.currentUserId/);
  assert.match(source, /emit\('remove', seat\.user_id\)/);
  assert.match(source, /emit\('restore', member\.user_id\)/);
  assert.match(source, /max-width:700px[\s\S]*grid-template-columns:1fr/);
});
