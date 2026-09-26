import { createHash } from 'node:crypto';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import postcss from 'postcss';
import cascadeLayers from '@csstools/postcss-cascade-layers';
import selectorParser from 'postcss-selector-parser';
import valueParser from 'postcss-value-parser';
import { transform } from 'lightningcss';

const root = new URL('../', import.meta.url);
const dist = new URL('../dist/', import.meta.url);
const compatRoot = 'html[data-css-compat="ready"]';

export function entryCssFiles(manifest, entryName) {
  const seen = new Set();
  const files = new Set();
  function visit(key) {
    if (seen.has(key)) return;
    seen.add(key);
    const entry = manifest[key];
    if (!entry) throw new Error(`Missing Vite manifest entry: ${key}`);
    for (const dependency of entry.imports || []) visit(dependency);
    for (const css of entry.css || []) files.add(css);
    for (const dependency of entry.dynamicImports || []) visit(dependency);
  }
  visit(entryName);
  const rank = (file) => file.includes('/style-') ? 0 : file.includes('/main-') ? 1 : 2;
  return [...files].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

export function mainCssFiles(manifest) {
  return entryCssFiles(manifest, 'index.html');
}

function replaceRuntimeColorMix(value) {
  const parsed = valueParser(value);
  parsed.walk((node) => {
    if (node.type !== 'function' || node.value !== 'color-mix') return;
    const variables = [];
    valueParser.walk(node.nodes, (child) => {
      if (child.type === 'function' && child.value === 'var') {
        variables.push(valueParser.stringify(child));
      }
    });
    if (!variables.length) return;
    node.type = 'word';
    node.value = variables[variables.length - 1];
    delete node.nodes;
  });
  return parsed.toString();
}

function lowerRegisteredProperties(css) {
  const defaults = new Map();
  css.walkAtRules('property', (rule) => {
    const initial = rule.nodes?.find((node) => node.prop === 'initial-value');
    if (initial) defaults.set(rule.params, initial.value);
    rule.remove();
  });
  css.walkDecls((declaration) => {
    const parsed = valueParser(replaceRuntimeColorMix(declaration.value));
    parsed.walk((node) => {
      if (node.type !== 'function' || node.value !== 'var') return;
      const name = node.nodes?.[0]?.value;
      if (!defaults.has(name) || node.nodes.some((part) => part.type === 'div')) return;
      node.nodes = valueParser(`${name}, ${defaults.get(name)}`).nodes;
    });
    declaration.value = parsed.toString();
  });
}

function scopeToCompat(css) {
  css.walkRules((rule) => {
    for (let parent = rule.parent; parent; parent = parent.parent) {
      if (parent.type === 'atrule' && /keyframes$/i.test(parent.name)) return;
    }
    const selectors = selectorParser().astSync(rule.selector);
    selectors.each((selector) => {
      const source = selector.toString();
      if (source.startsWith(compatRoot)) return;
      const scoped = /^(?:html|:root)(?=[\s.#:[>+~]|$)/.test(source)
        ? source.replace(/^(?:html|:root)/, compatRoot)
        : `${compatRoot} ${source}`;
      selector.replaceWith(selectorParser().astSync(scoped).first);
    });
    rule.selector = selectors.toString();
  });
}

async function buildEntryCompat(files, structuralFallback) {
  const sources = await Promise.all(files.map((file) => readFile(new URL(file, dist), 'utf8')));
  const css = postcss.parse(sources.join('\n'));
  lowerRegisteredProperties(css);
  const flattened = await postcss([cascadeLayers()]).process(css, { from: undefined });
  if (flattened.warnings().length) {
    throw new Error(flattened.warnings().map((warning) => warning.text).join('\n'));
  }
  const lowered = transform({
    filename: 'render-compat.css',
    code: Buffer.from(flattened.css),
    targets: { chrome: 70 << 16, safari: 12 << 16 },
    minify: true,
  });
  const output = postcss.parse(lowered.code.toString());
  output.walkAtRules('supports', (rule) => {
    if (/color-mix\(|oklch\(/.test(rule.params)) rule.remove();
  });
  scopeToCompat(output);
  const cssText = `${output.toString()}\n${structuralFallback}`;
  const unsupported = /@layer\b|@property\b|color-mix\(/.exec(cssText);
  if (unsupported) {
    throw new Error(`Generated compatibility CSS contains ${unsupported[0]} near ${cssText.slice(Math.max(0, unsupported.index - 80), unsupported.index + 120)}`);
  }
  const hash = createHash('sha256').update(cssText).digest('hex').slice(0, 12);
  const filename = `render-compat-${hash}.css`;
  const destination = new URL('compat/', dist);
  await mkdir(destination, { recursive: true });
  await writeFile(new URL(filename, destination), cssText);
  return { filename, files, cssText };
}

export async function buildRenderCompat() {
  const manifest = JSON.parse(await readFile(new URL('.vite/manifest.json', dist), 'utf8'));
  const structuralFallback = await readFile(new URL('public/compat/render-compat.css', root), 'utf8');
  const entries = [
    { name: 'index.html', marker: '<script src="/compat/render-compat.js?v=2"></script>', files: mainCssFiles(manifest) },
    { name: 'human/index.html', marker: '<script src="/compat/render-compat.js?v=3"></script>', files: entryCssFiles(manifest, 'human/index.html') },
  ];
  if (!entries[0].files.length || !entries[0].files.some((file) => file.includes('/style-'))) {
    throw new Error('Main CSS was not found in the Vite manifest');
  }
  if (!entries[1].files.length || !entries[1].files.some((file) => file.includes('/human-'))) {
    throw new Error('Play CSS was not found in the Vite manifest');
  }
  for (const entry of entries) {
    const built = await buildEntryCompat(entry.files, structuralFallback);
    entry.filename = built.filename;
    entry.cssText = built.cssText;
    const htmlFile = new URL(entry.name, dist);
    const html = await readFile(htmlFile, 'utf8');
    if (!html.includes(entry.marker)) throw new Error(`${entry.name} compatibility loader was not found`);
    await writeFile(htmlFile, html.replace(entry.marker,
      `<script>window.__RENDER_COMPAT_CSS_URL__="/compat/${entry.filename}";</script>\n    ${entry.marker}`));
  }
  return { filename: entries[0].filename, files: entries[0].files, entries };
}
