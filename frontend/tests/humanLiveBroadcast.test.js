import test from 'node:test';
import assert from 'node:assert/strict';
import { copyLiveShareText, liveShareText } from '../src/human/liveShareText.js';
import { liveAppearanceTileStyle } from '../src/human/liveAppearance.js';
import { liveEmptyTileColors } from '../src/live/tilePalette.js';

test('player stream share text includes variant, current score and room URL', () => {
  const context = { run: { variant: '3x4', score: 36123 } };
  const room = { url: 'https://live.2048tables.online/rooms/h-example' };
  const zh = liveShareText(context, room, 'zh');
  const en = liveShareText(context, room, 'en');
  for (const value of [zh, en]) {
    assert.match(value, /3×4/);
    assert.match(value, /36,123/);
    assert.match(value, /https:\/\/live\.2048tables\.online\/rooms\/h-example/);
  }
});

test('player stream appearance resolves streamer colors independently of viewer palette', () => {
  const appearance = { version: 1, empty: { background: '#010203', color: '#040506' },
    tiles: { 2: { background: '#112233', color: '#ddeeff' } } };
  assert.deepEqual(liveAppearanceTileStyle(appearance, 0), { background: '#010203', color: '#040506' });
  assert.deepEqual(liveAppearanceTileStyle(appearance, 2), { background: '#112233', color: '#ddeeff' });
  assert.equal(liveAppearanceTileStyle(appearance, 4), null);
});

test('Live empty cells use the Live surface palette', () => {
  assert.deepEqual(liveEmptyTileColors(), {
    background: 'var(--color-empty)',
    color: 'var(--text-secondary)',
  });
});

test('sharing a player stream copies text without invoking the system share sheet', async () => {
  let copied = '', shared = false;
  const platform = { share() { shared = true; throw Error('must not open share sheet'); }, clipboard: { async writeText(value) { copied = value; } } };
  assert.equal(await copyLiveShareText('watch me', platform, null), true);
  assert.equal(copied, 'watch me');
  assert.equal(shared, false);
});
