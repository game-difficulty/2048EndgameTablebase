import assert from 'node:assert/strict';
import test from 'node:test';

import { loadGuideDocument, loadGuideIndex } from '../src/features/help/services/guideLibrary.js';

function jsonResponse(data, url) {
  return {
    ok: true,
    status: 200,
    url,
    async json() {
      return data;
    },
  };
}

test('loads guide registry entries and resolves document URLs', async () => {
  let requestOptions;
  const entries = await loadGuideIndex(async (_url, options) => {
    requestOptions = options;
    return jsonResponse({
      documents: [{
        id: 'guide-a',
        language: 'zh',
        source: 'guide-a/document.json',
        title: 'Guide A',
      }],
    }, 'https://example.test/app/guides/index.json');
  }, 'https://example.test/app/guides/index.json');

  assert.equal(entries.length, 1);
  assert.equal(entries[0].documentUrl, 'https://example.test/app/guides/guide-a/document.json');
  assert.equal(requestOptions.cache, 'no-cache');
});

test('resolves original figure assets relative to the guide document', async () => {
  const document = await loadGuideDocument({
    documentUrl: 'https://example.test/app/guides/guide-a/document.json',
  }, async () => jsonResponse({
    id: 'guide-a',
    toc: [],
    blocks: [
      { type: 'paragraph', text: 'Text' },
      { type: 'figure', src: 'media/img_0001.png', boards: [] },
    ],
  }, 'https://example.test/app/guides/guide-a/document.json'));

  assert.equal(
    document.blocks[1].src,
    'https://example.test/app/guides/guide-a/media/img_0001.png',
  );
});
