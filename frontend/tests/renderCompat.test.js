import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import postcss from 'postcss';

const source = readFileSync(new URL('../public/compat/render-compat.js', import.meta.url), 'utf8');
const css = readFileSync(new URL('../public/compat/render-compat.css', import.meta.url), 'utf8');
function detect({ layers = true, unsupported = [], registered = true, missingCss = false, brokenProbe = false } = {}) {
  const attrs = {}, writes = [], children = new Set();
  const parent = {
    appendChild(el) { children.add(el); el.parentNode = this; },
    removeChild(el) { children.delete(el); el.parentNode = null; },
    setAttribute(k, v) { attrs[k] = v; },
  };
  const window = {
    CSS: missingCss ? undefined : { supports: p => !unsupported.includes(p), registerProperty: registered ? () => {} : undefined },
    getComputedStyle: () => { if (brokenProbe) throw Error('probe'); return { width: layers ? '13px' : '0px' }; },
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
for (const options of [{ layers: false }, { unsupported: ['color'] }, { unsupported: ['aspect-ratio'] }, { registered: false }, { missingCss: true }, { brokenProbe: true }]) {
  test(`unsupported capabilities load exactly one stylesheet: ${JSON.stringify(options)}`, () => {
    const actual = detect(options);
    assert.equal(actual.result.mode, 'compat');
    assert.ok(actual.result.reasons.length);
    assert.equal(actual.writes.length, 1);
    assert.match(actual.writes[0], /render-compat\.css\?v=1/);
    assert.match(actual.writes[0], /onerror=/);
    assert.equal(actual.children.size, 0);
  });
}
test('both entry documents run the classic detector before the application', () => {
  for (const path of ['../index.html', '../live/index.html']) {
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
  for (const selector of ['.board-stage', '.tile-inner', '.fixed.inset-0', '.account-popover', '.chat-list']) assert.ok(css.includes(selector));
});
