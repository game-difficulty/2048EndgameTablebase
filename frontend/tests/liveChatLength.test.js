import test from 'node:test';
import assert from 'node:assert/strict';
import { chatLength, CHAT_LIMIT } from '../src/live/chatLength.js';

test('weighted chat accepts 80 Latin or 40 CJK characters, including mixed text', () => {
  assert.equal(CHAT_LIMIT, 80);
  for (const text of ['a'.repeat(80), '中'.repeat(40), '中'.repeat(20)+'a'.repeat(40)]) {
    assert.equal(chatLength(text), 80);
  }
  assert.equal(chatLength('hello 中文!'), 11);
});
test('joined emoji, flags, modifiers and combining accents match the server', () => {
  for (const text of ['😀', '👍🏽', '👨‍👩‍👧‍👦', '🇨🇳', '❤️']) assert.equal(chatLength(text), 2);
  assert.equal(chatLength('e\u0301'), 1);
});
