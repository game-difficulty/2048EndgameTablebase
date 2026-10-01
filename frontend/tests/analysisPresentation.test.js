import test from 'node:test';
import assert from 'node:assert/strict';
import { analysisScoreLabel, formatAnalysisFit, replayDisplayName } from '../src/features/replay/analysisPresentation.js';

test('internal replay IDs are replaced while meaningful names survive', () => {
  assert.equal(replayDisplayName('6650a9c9-0084-53e6-b89e-147220dde883.vrs', 'Game 1'), 'Game 1');
  assert.equal(replayDisplayName('C:\\tmp\\01_6650a9c9-0084-53e6-b89e-147220dde883.vrs', 'Game 1'), 'Game 1');
  assert.equal(replayDisplayName('record-32768.vrs', 'Game 1'), 'record-32768.vrs');
});

test('accuracy distinguishes missing data from zero and perfect results', () => {
  assert.equal(formatAnalysisFit(null), '—');
  assert.equal(formatAnalysisFit(0), '0.0%');
  assert.equal(formatAnalysisFit(0.988), '98.8%');
  assert.equal(formatAnalysisFit(1), '100.0%');
});

test('score labels never fabricate a score for missing metadata', () => {
  assert.equal(analysisScoreLabel(null), '');
  assert.equal(analysisScoreLabel(0), '0 分');
  assert.equal(analysisScoreLabel(513912, 'en'), '513,912 pts');
});
