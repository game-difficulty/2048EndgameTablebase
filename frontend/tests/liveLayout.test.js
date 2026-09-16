import test from 'node:test';
import assert from 'node:assert/strict';
import { livePageScale, livePageWidth } from '../src/live/liveLayout.js';

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

test('extremely wide windows letterbox at two-to-one instead of stretching the columns', () => {
  for (const [width, height] of [[1500,100], [2560,200], [1920,720]]) {
    const visualWidth = livePageWidth(width, height) * livePageScale(width, height);
    assert.ok(Math.abs(visualWidth - height * 2) < 1e-6);
    assert.ok(visualWidth <= width);
  }
  for (const [width, height] of [[1616,925], [1260,922], [390,844]]) {
    assert.ok(Math.abs(livePageWidth(width, height) * livePageScale(width, height) - width) < 1e-6);
  }
});
