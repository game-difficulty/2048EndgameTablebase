import test from 'node:test';
import assert from 'node:assert/strict';
import { emphasisParts } from '../src/features/billing/tokenSourceText.js';

test('emphasis preserves text and punctuation', () => {
  assert.deepEqual(emphasisParts('**Source**: earn **1,000**.'), [
    {bold:true,text:'Source'}, {bold:false,text:': earn '},
    {bold:true,text:'1,000'}, {bold:false,text:'.'},
  ]);
});
test('plain text and incomplete markers stay literal', () => {
  assert.deepEqual(emphasisParts('plain **unfinished'), [{bold:false,text:'plain **unfinished'}]);
  assert.deepEqual(emphasisParts('**<script>**'), [{bold:true,text:'<script>'}]);
});
