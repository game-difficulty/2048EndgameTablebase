import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import postcss from 'postcss';
import { entryCssFiles, mainCssFiles } from '../scripts/build-render-compat.mjs';

const source = readFileSync(new URL('../public/compat/render-compat.js', import.meta.url), 'utf8');
const css = readFileSync(new URL('../public/compat/render-compat.css', import.meta.url), 'utf8');
const humanCss = readFileSync(new URL('../src/human/human.css', import.meta.url), 'utf8');
const accountPopover = readFileSync(new URL('../src/features/auth/AccountPopover.vue', import.meta.url), 'utf8');
function detect({ layers = true, unsupported = [], registered = true, missingCss = false, brokenProbe = false, generatedUrl = '' } = {}) {
  const attrs = {}, writes = [], children = new Set();
  const parent = {
    appendChild(el) { children.add(el); el.parentNode = this; },
    removeChild(el) { children.delete(el); el.parentNode = null; },
    setAttribute(k, v) { attrs[k] = v; },
  };
  const window = {
    CSS: missingCss ? undefined : { supports: p => !unsupported.includes(p), registerProperty: registered ? () => {} : undefined },
    getComputedStyle: () => { if (brokenProbe) throw Error('probe'); return { width: layers ? '13px' : '0px' }; },
    __RENDER_COMPAT_CSS_URL__: generatedUrl,
  };
  const document = { documentElement: parent, head: parent, createElement: () => ({ style: {} }), write: html => writes.push(html) };
  runInNewContext(source, { window, document });
  return { attrs, writes, children, result: window.__RENDER_COMPAT__ };
}
test('modern engines never load the compatibility stylesheet', () => {
  const actual = detect();
  assert.equal(actual.result.mode, 'modern');
  assert.equal(actual.writes.length, 0);
  assert.equal(actual.attrs['data-css-compat'], 'modern');
  assert.equal(actual.children.size, 0);
});
for (const options of [{ layers: false }, { unsupported: ['color'] }, { unsupported: ['aspect-ratio'] }, { unsupported: ['grid-template-rows'] }, { registered: false }, { missingCss: true }, { brokenProbe: true }]) {
  test(`unsupported capabilities load exactly one stylesheet: ${JSON.stringify(options)}`, () => {
    const actual = detect(options);
    assert.equal(actual.result.mode, 'compat');
    assert.ok(actual.result.reasons.length);
    assert.equal(actual.writes.length, 1);
    assert.match(actual.writes[0], /render-compat\.css\?v=3/);
    assert.match(actual.writes[0], /onerror=/);
    assert.equal(actual.children.size, 0);
  });
}
test('built main page loads its generated compatibility CSS', () => {
  const actual = detect({ layers: false, generatedUrl: '/compat/render-compat-123.css' });
  assert.equal(actual.writes.length, 1);
  assert.match(actual.writes[0], /render-compat-123\.css/);
  assert.doesNotMatch(actual.writes[0], /render-compat\.css\?v=3/);
});
test('build keeps main and Play compatibility styles in separate bundles', () => {
  const manifest = {
    'index.html': { css: ['assets/main-123.css'], dynamicImports: ['game', 'battle', 'settings'] },
    'human/index.html': { css: ['assets/human.css'], dynamicImports: ['human-profile'] },
    game: { css: ['assets/game.css'], imports: ['shared'], dynamicImports: ['game-dialog'] },
    battle: { css: ['assets/battle.css'], imports: ['shared'] },
    settings: { css: ['assets/settings.css'] },
    'human-profile': { css: ['assets/human-profile.css'] },
    'game-dialog': { css: ['assets/dialog.css'] },
    shared: { css: ['assets/style-shared.css'] },
    'live/index.html': { css: ['assets/live.css'] },
  };
  assert.deepEqual(mainCssFiles(manifest), [
    'assets/style-shared.css',
    'assets/main-123.css',
    'assets/battle.css',
    'assets/dialog.css',
    'assets/game.css',
    'assets/settings.css',
  ]);
  assert.deepEqual(entryCssFiles(manifest, 'human/index.html'), [
    'assets/human-profile.css',
    'assets/human.css',
  ]);
  assert.throws(() => mainCssFiles({ 'index.html': { dynamicImports: ['missing'] } }), /Missing Vite manifest entry/);
});
test('both entry documents run the classic detector before the application', () => {
  for (const path of ['../index.html', '../live/index.html', '../human/index.html']) {
    const html = readFileSync(new URL(path, import.meta.url), 'utf8');
    assert.ok(html.indexOf('/compat/render-compat.js') < html.indexOf('type="module"'));
  }
});
test('compatibility CSS is unlayered and every selector is opt-in', () => {
  const root = postcss.parse(css);
  root.walkAtRules(rule => assert.fail(`Unexpected modern CSS at-rule: ${rule.name}`));
  root.walkRules(rule => {
    for (const selector of rule.selectors) assert.ok(selector.startsWith('html[data-css-compat="ready"]'));
  });
  assert.doesNotMatch(css, /color-mix\(|oklch\(|:has\(/);
  for (const selector of ['.board-stage', '.tile-inner', '.fixed.inset-0', '.account-popover', '.chat-list', '.game-column', '.human-board .tile']) assert.ok(css.includes(selector));
  assert.match(css, /\.modal \{ max-height:90vh; \}/);
  assert.match(css, /\.analysis-poster-dialog \{ max-height:95vh; \}/);
  assert.match(css, /\.human-account-control \.account-popover \{ max-height:calc\(100vh - 4\.5rem\); \}/);
});
test('Play overlays retain viewport-height fallbacks before dynamic viewport units', () => {
  assert.match(humanCss, /max-height:\s*90vh;\s*max-height:\s*90dvh/);
  assert.match(humanCss, /max-height:\s*95vh;\s*max-height:\s*95dvh/);
  assert.match(accountPopover, /max-height:calc\(100vh - 4\.5rem\);max-height:calc\(100dvh - 4\.5rem\)/);
});
