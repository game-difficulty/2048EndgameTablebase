import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const humanRoot = new URL('../src/human/', import.meta.url);
const humanRootPath = fileURLToPath(humanRoot);
const i18nSource = readFileSync(new URL('i18n.js', humanRoot), 'utf8');
const dictionaryBody = i18nSource.split('const en = {')[1].split('\n};')[0];
const englishKeys = new Set([...dictionaryBody.matchAll(/(['"])((?:\\.|(?!\1).)*?)\1\s*:/gs)].map(match => match[2]));

function sourceFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(path);
    return ['.js', '.vue'].includes(extname(entry.name)) && entry.name !== 'i18n.js' ? [path] : [];
  });
}

test('literal Chinese text passed to the human-site translator has an English entry', () => {
  const missing = [];
  for (const file of sourceFiles(humanRootPath)) {
    const source = readFileSync(file, 'utf8');
    for (const match of source.matchAll(/(?<![\w$])t\(\s*(['"])((?:\\.|(?!\1).)*?)\1\s*\)/gs)) {
      const key = match[2].trim();
      if (/[\u3400-\u9fff]/.test(key) && !englishKeys.has(key)) missing.push(`${file}: ${key}`);
    }
  }
  assert.deepEqual(missing, []);
});

test('session and profile errors shown through dynamic popup text have English entries', () => {
  const required = [
    '对局写入权已变化，请重新检查本地对局。',
    '登录已失效，请重新登录后检查本地进度。',
    '无法生成分享图。',
  ];
  assert.deepEqual(required.filter(key => !englishKeys.has(key)), []);
});

test('bilingual analysis dialog labels supply a real English alternative', () => {
  const files = ['HumanAnalysisDialog.vue', 'HumanAnalysisPosterDialog.vue'];
  const invalid = [];
  for (const name of files) {
    const source = readFileSync(new URL(name, humanRoot), 'utf8');
    for (const match of source.matchAll(/label\(\s*'([^']+)'\s*,\s*'([^']*)'\s*\)/g)) {
      if (!match[2].trim() || /[\u3400-\u9fff]/.test(match[2])) invalid.push(`${name}: ${match[1]}`);
    }
  }
  assert.deepEqual(invalid, []);
});
