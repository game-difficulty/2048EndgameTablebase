import test from 'node:test';
import assert from 'node:assert/strict';
import { liveTileColors, liveBoardPalette, setLiveTilePalette } from '../src/live/tilePalette.js';
import { readSharedTilePalette, writeSharedTilePalette } from '../src/utils/sharedTilePalette.js';

test('live board and compact badges share background and text for every tile', () => {
  const board = liveBoardPalette();
  for (let exponent = 1; exponent <= 31; exponent++) {
    const value = 2 ** exponent;
    assert.equal(board[`--color-tile-${value}`], liveTileColors(value).background);
    assert.equal(board[`--color-text-${value}`], liveTileColors(value).color);
  }
});

test('high tiles have distinct colored defaults instead of black', () => {
  for (const [value, color] of [[4096, '#8056b3'], [8192, '#6652a3'], [16384, '#4c5498'], [32768, '#287c8e'], [65536, '#247a60']]) {
    assert.equal(liveTileColors(value).background, color);
    assert.equal(liveTileColors(value).color, '#f9f6f2');
  }
});

test('missing, partial and malformed themes always retain a complete palette', () => {
  for (const theme of [null, [], [null, {}, 123, 'bad', 'var(--missing)'], ['#abcdef']]) {
    setLiveTilePalette(theme);
    for (let exponent=1;exponent<=31;exponent++) {
      const colors=liveTileColors(2**exponent);
      assert.match(colors.background,/^#[0-9a-f]{6}$/i);
      assert.notEqual(colors.background,'#000000');
      assert.match(colors.color,/^#[0-9a-f]{6}$/i);
    }
  }
  setLiveTilePalette([{background:'#123456',color:'#ffffff'}]);
  assert.deepEqual(liveTileColors(2),{background:'#123456',color:'#ffffff'});
  setLiveTilePalette(null);
});

test('unreadable or corrupt palette cookies return the default selection', () => {
  const previous=Object.getOwnPropertyDescriptor(globalThis,'document');
  try {
    for (const cookie of ['', '2048tables-tile-palette=%broken', '2048tables-tile-palette=null']) {
      globalThis.document={cookie};assert.equal(readSharedTilePalette(),null);
    }
    globalThis.document={get cookie(){throw new Error('storage blocked');}};
    assert.equal(readSharedTilePalette(),null);
  } finally {
    if(previous)Object.defineProperty(globalThis,'document',previous);else delete globalThis.document;
  }
});

test('legacy all-black cookies cannot override the default palette after refresh', () => {
  const previous=Object.getOwnPropertyDescriptor(globalThis,'document');
  try {
    for (const colors of [Array(36).fill('#000000'),Array(36).fill({background:'#000000',color:'#f9f6f2'})]) {
      globalThis.document={cookie:`2048tables-tile-palette=${encodeURIComponent(JSON.stringify(colors))}`};
      assert.equal(readSharedTilePalette(),null);
      setLiveTilePalette(colors);
      assert.equal(liveTileColors(2).background,'#eee4da');
      assert.equal(liveTileColors(65536).background,'#247a60');
      const cookie=document.cookie;
      writeSharedTilePalette(colors);
      assert.equal(document.cookie,cookie);
    }
    setLiveTilePalette(['#000000','#abcdef']);
    assert.equal(liveTileColors(2).background,'#000000');
    assert.equal(liveTileColors(4).background,'#abcdef');
  } finally {
    setLiveTilePalette(null);
    if(previous)Object.defineProperty(globalThis,'document',previous);else delete globalThis.document;
  }
});
