import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse, compileStyle } from '@vue/compiler-sfc';

test('practice uses shared responsive typography and readable wrapping stats', () => {
  const source = readFileSync(new URL('../src/projects/ProjectPlayground.vue', import.meta.url), 'utf8');
  const { descriptor, errors } = parse(source);
  assert.deepEqual(errors, []);
  const result = compileStyle({ source: descriptor.styles[0].content, filename: 'ProjectPlayground.vue', id: 'practice-type', scoped: true });
  assert.deepEqual(result.errors, []);
  const css = descriptor.styles[0].content;
  assert.match(css, /\.project-heading h1\{font-size:clamp\(24px,2\.3vw,30px\)/);
  assert.match(css, /\.rules-panel h2\{font-size:18px/);
  assert.match(css, /\.project-lab \.game-hud\{display:grid;grid-template-columns:repeat\(auto-fit/);
  assert.match(css, /\.game-hud>div\{min-width:0;width:auto/);
  assert.match(css, /\.game-hud strong\{[^}]*white-space:nowrap/);
  assert.doesNotMatch(css, /:lang\(|\[lang[=|]|\.is-english|\.lang-en/);
});
