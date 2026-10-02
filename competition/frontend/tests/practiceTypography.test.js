import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parse, compileStyle } from '@vue/compiler-sfc';
import { ALL_PROJECTS, PROJECT_BY_ID, PROJECT_BY_ORDER, PRACTICE_PROJECTS } from '../src/projects/catalog.js';

test('all 20 project routes share the practice page, independent of menu visibility', () => {
  const source = readFileSync(new URL('../src/projects/ProjectPlayground.vue', import.meta.url), 'utf8');
  assert.match(source, /const project = computed\(\(\) => PROJECT_BY_ID\[props\.projectId\] \|\| null\)/);
  assert.equal(ALL_PROJECTS.length, 20);
  for (const project of ALL_PROJECTS) {
    assert.equal(PROJECT_BY_ID[project.id], project);
    assert.equal(PROJECT_BY_ORDER[project.order], project);
    assert.equal(project.practicePath, `/practice/${project.order}`);
  }
  assert.equal(PRACTICE_PROJECTS.length, 12, 'preserve the current menu');
});

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

test('practice controls mobile text inflation without disabling zoom or language-specific navigation', () => {
  const source = readFileSync(new URL('../src/projects/ProjectPlayground.vue', import.meta.url), 'utf8');
  const globalCss = readFileSync(new URL('../src/styles.css', import.meta.url), 'utf8');
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  assert.match(source, /\.project-lab\{-webkit-text-size-adjust:100%;text-size-adjust:100%\}/);
  assert.match(source, /\.lab-header nav\{width:100%;gap:10px;flex-wrap:wrap/);
  assert.doesNotMatch(globalCss, /html\[lang=en\] \.lab-header/);
  assert.doesNotMatch(html, /user-scalable\s*=\s*no|maximum-scale\s*=\s*1/);
});
