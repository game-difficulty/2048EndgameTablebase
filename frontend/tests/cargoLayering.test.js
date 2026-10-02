import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('all cargo shapes cover port guides without hiding the labels in both renderers', () => {
  const renderers = [
    ['../../competition/frontend/src/projects/CargoBoard.vue', '.cargo-port', '.cargo-port span', '.cargo-piece-box', '.cargo-port b'],
    ['../src/live/content/CargoTransportView.vue', '.live-port', '.live-port i', '.special-cargo', '.live-port span'],
  ];
  for (const [path, port, guide, cargo, label] of renderers) {
    const source = readFileSync(new URL(path, import.meta.url), 'utf8');
    const rule = selector => {
      const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const matches = [...source.matchAll(new RegExp(`${escaped}\\{([^}]+)\\}`, 'g'))];
      assert.ok(matches.length, `Missing ${selector} in ${path}`);
      return matches.map(match => match[1]).join(';');
    };
    assert.doesNotMatch(rule(port), /z-index|transform|opacity|isolation/, 'Port must not trap guides and labels in one stacking context');
    const z = selector => Number(rule(selector).match(/z-index:(\d+)/)?.[1]);
    assert.ok(z(guide) < z(cargo), 'Guides must be underneath every cargo shape');
    assert.ok(z(cargo) < z(label), 'Labels must remain above cargo');
  }
});

test('practice, competition and current live view share the corrected cargo renderer', () => {
  for (const path of ['../../competition/frontend/src/projects/ProjectPlayground.vue', '../../competition/frontend/src/App.vue', '../../competition/shared/ObservedProjectBoard.vue']) {
    assert.match(readFileSync(new URL(path, import.meta.url), 'utf8'), /import CargoBoard from/);
  }
  assert.match(readFileSync(new URL('../src/live/content/projectViewRegistry.js', import.meta.url), 'utf8'), /'cargo-transport\|cargo-transport-v1': StreamProjectView/);
});
