import test from 'node:test';
import assert from 'node:assert/strict';
import { livePageScale } from '../src/live/liveLayout.js';

test('narrow windows retain the same minimum layout width through uniform scaling', () => {
  for (const width of [320,390,800,1000,1200]) {
    assert.equal(width/livePageScale(width),1500);
  }
});
test('wide screens keep the chosen 85 percent scale', () => {
  for(const width of [1280,1920,2560]) assert.equal(livePageScale(width),.85);
});

test('short windows scale the fixed layout to preserve the gift bar and chat input', () => {
  for (const [width, height] of [[1920,720],[1616,925],[1280,600],[844,390],[390,844]]) {
    const scale = livePageScale(width, height);
    assert.ok(scale <= .85);
    assert.ok(width / scale >= 1500 - 1e-6);
    assert.ok(height / scale >= 1080 - 1e-6);
  }
});
