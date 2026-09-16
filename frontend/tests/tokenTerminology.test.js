import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

test('frontend copy does not reintroduce retired balance names', () => {
  const root = fileURLToPath(new URL('../src/', import.meta.url));
  function check(directory) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) check(path);
      else if (/\.(vue|js|json)$/.test(entry.name)) {
        const source = readFileSync(path, 'utf8');
        assert.doesNotMatch(source, /充值余额|充值额度|付费余额|付费\s*token|常驻余额|\bpaid\s+(?:token\s+)?balance\b|\bpaid\s+tokens?\b/i, path);
      }
    }
  }
  check(root);
});
