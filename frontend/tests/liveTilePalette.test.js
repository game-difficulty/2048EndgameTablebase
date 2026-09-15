import test from 'node:test';
import assert from 'node:assert/strict';
import { liveTileColors, liveBoardPalette } from '../src/live/tilePalette.js';

test('live board and compact badges share background and text for every tile', () => {
  const board = liveBoardPalette();
  for (let exponent = 1; exponent <= 31; exponent++) {
    const value = 2 ** exponent;
    assert.equal(board[`--color-tile-${value}`], liveTileColors(value).background);
    assert.equal(board[`--color-text-${value}`], liveTileColors(value).color);
  }
});

test('purple milestones remain distinct while 32K and 65K stay black', () => {
  for (const [value, color] of [[4096, '#9100cf'], [8192, '#590080'], [16384, '#36004d'], [32768, '#000000'], [65536, '#000000']]) {
    assert.equal(liveTileColors(value).background, color);
    assert.equal(liveTileColors(value).color, '#f9f6f2');
  }
});
