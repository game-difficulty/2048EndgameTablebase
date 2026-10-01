import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('Cao lettering scales with the special block in both cargo renderers', () => {
  for (const path of ['../../competition/frontend/src/projects/CargoBoard.vue', '../src/live/content/CargoTransportView.vue']) {
    const source = readFileSync(new URL(path, import.meta.url), 'utf8');
    assert.match(source, /\.cargo-art\.cao-cargo\{[^}]*container-type:inline-size|\.special-cargo\.cao-cargo\{[^}]*container-type:inline-size/);
    assert.match(source, /\.cao-cargo strong\{[^}]*font-size:60cqw/);
  }
});
