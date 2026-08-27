import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import test from 'node:test';

import { MinigameController } from '../src/features/minigames/engine/controller.js';
import { MinigameRankedRecorder } from '../src/features/minigames/engine/rankedRecorder.js';
import { createMinigameRuntime } from '../src/features/minigames/engine/runtime.js';
import { flattenBoard } from '../src/features/minigames/engine/utils.js';
import {
  decodeMgo1,
  encodeMgo1,
  MGO1_END_REASON,
} from '../src/features/minigames/protocol/index.js';

const SEED = '0123456789abcdeffedcba9876543210';
const RUN_ID = '123e4567-e89b-42d3-a456-426614174000';

async function completedRecord() {
  const runtime = createMinigameRuntime({ seedHex: SEED, clock: { now: () => 1000 } });
  let recorder;
  const controller = new MinigameController({
    difficulty: 1,
    runtime,
    onOperation({ operation, atMs, state }) {
      recorder.record(operation, atMs, state);
      if (state.engine?.isOver) recorder.finish(atMs, state);
    },
  });
  recorder = new MinigameRankedRecorder({
    runId: RUN_ID,
    gameId: 'design-master-1',
    difficulty: 1,
    seedHex: SEED,
    startedAtMs: 1000,
  });
  await controller.startGame('design-master-1', null, runtime);
  const directions = ['left', 'down', 'right', 'up'];
  for (let index = 0; !controller.engine.isOver && index < 20000; index += 1) {
    await controller.move(directions[index % directions.length]);
  }
  assert.equal(controller.engine.isOver, true);
  return {
    record: recorder.encode(),
    expected: {
      run_id: RUN_ID,
      seed_hex: SEED,
      rules_version: 1,
      game_id: 'design-master-1',
      difficulty: 1,
      score: controller.engine.score,
      trophy_tier: controller.engine.isPassed,
      highest_tile_exp: controller.engine.highestTileExp,
      final_board: flattenBoard(controller.engine.board),
      board_rows: controller.engine.rows,
      board_cols: controller.engine.cols,
      action_count: recorder.mutableActionCount,
      elapsed_ms: 0,
    },
  };
}

async function retiredRecord() {
  const runtime = createMinigameRuntime({ seedHex: SEED, clock: { now: () => 1000 } });
  let recorder;
  const controller = new MinigameController({
    difficulty: 1,
    runtime,
    onOperation({ operation, atMs, state }) {
      recorder.record(operation, atMs, state);
    },
  });
  recorder = new MinigameRankedRecorder({
    runId: RUN_ID,
    gameId: 'column-chaos',
    difficulty: 1,
    seedHex: SEED,
    startedAtMs: 1000,
  });
  await controller.startGame('column-chaos', null, runtime);
  await controller.move('left');
  recorder.finish(1000, null, MGO1_END_REASON.RETIRED);
  return {
    record: recorder.encode(),
    expected: {
      run_id: RUN_ID,
      seed_hex: SEED,
      rules_version: 1,
      game_id: 'column-chaos',
      difficulty: 1,
      score: controller.engine.score,
      trophy_tier: controller.engine.isPassed,
      highest_tile_exp: controller.engine.highestTileExp,
      final_board: flattenBoard(controller.engine.board),
      board_rows: controller.engine.rows,
      board_cols: controller.engine.cols,
      action_count: recorder.mutableActionCount,
      elapsed_ms: 0,
    },
  };
}

const runVerifier = (request) => new Promise((resolve, reject) => {
  const child = spawn(process.execPath, ['scripts/minigameVerifier.mjs'], {
    cwd: new URL('..', import.meta.url),
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  let stdout = '';
  let stderr = '';
  child.stdout.setEncoding('utf8');
  child.stderr.setEncoding('utf8');
  child.stdout.on('data', (chunk) => { stdout += chunk; });
  child.stderr.on('data', (chunk) => { stderr += chunk; });
  child.on('error', reject);
  child.on('close', (code) => {
    if (code !== 0) reject(new Error(stderr || `Verifier exited with ${code}.`));
    else resolve(JSON.parse(stdout.trim()));
  });
  child.stdin.end(`${JSON.stringify(request)}\n`);
});

test('Node verifier accepts an honest compact operation stream', async () => {
  const { record, expected } = await completedRecord();
  const response = await runVerifier({ request_id: 'test-1', record_encoding: record, expected });
  assert.equal(response.ok, true);
  assert.equal(response.result.score, expected.score);
});

test('Node verifier accepts a retired trophy checkpoint stream', async () => {
  const { record, expected } = await retiredRecord();
  const response = await runVerifier({ request_id: 'test-retired', record_encoding: record, expected });
  assert.equal(response.ok, true);
  assert.equal(response.result.score, expected.score);
});

test('Node verifier rejects a forged claimed score', async () => {
  const { record, expected } = await completedRecord();
  const response = await runVerifier({
    request_id: 'test-2',
    record_encoding: record,
    expected: { ...expected, score: expected.score + 4 },
  });
  assert.equal(response.ok, false);
  assert.match(response.error, /score_mismatch/u);
});

test('Node verifier rejects a valid-CRC stream with a changed operation', async () => {
  const { record, expected } = await completedRecord();
  const decoded = decodeMgo1(record);
  const firstMove = decoded.actions.find((action) => action.type === 'move');
  firstMove.direction = firstMove.direction === 'left' ? 'right' : 'left';
  const tampered = encodeMgo1(decoded);
  const response = await runVerifier({
    request_id: 'test-3',
    record_encoding: tampered,
    expected,
  });
  assert.equal(response.ok, false);
  assert.match(response.error, /mismatch/u);
});
