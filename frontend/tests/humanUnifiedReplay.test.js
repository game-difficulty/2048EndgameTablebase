import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const humanApp = readFileSync(new URL('../src/human/HumanApp.vue', import.meta.url), 'utf8');
const replayApp = readFileSync(new URL('../public/verse-replay/app.js', import.meta.url), 'utf8');
const replayStyles = readFileSync(new URL('../public/verse-replay/styles.css', import.meta.url), 'utf8');

test('archived and current games use the unified replay viewer', () => {
  assert.match(humanApp, /new URL\('\/verse-replay\/'/);
  assert.match(humanApp, /searchParams\.set\('human-run', id\)/);
  assert.match(humanApp, /openReplayViewer\(replay, language\.value\)/);
  assert.doesNotMatch(humanApp, /view === 'replay'/);
  assert.doesNotMatch(humanApp, /local-replay/);
  assert.doesNotMatch(humanApp, /buildReplay|parseReplay/);
});

test('unified viewer sizes the timeline from each replay board ratio', () => {
  assert.match(replayApp, /--board-aspect-ratio', replay\.width \/ replay\.height/);
  assert.match(
    replayStyles,
    /grid-template-rows:\s*calc\(\(min\(96vw, 820px\) - 256px\) \/ var\(--board-aspect-ratio\)\) auto/,
  );
  assert.match(replayStyles, /\.timeline-panel\s*\{[^}]*height:\s*100%/s);
  assert.match(replayStyles, /grid-template-rows:\s*auto auto auto/);
});
