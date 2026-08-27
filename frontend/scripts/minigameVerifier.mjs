import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

import createEvilCore from '../public/wasm/evil_core.js';
import { replayMgo1 } from '../src/features/minigames/engine/rankedReplay.js';
import { flattenBoard } from '../src/features/minigames/engine/utils.js';

let evilModulePromise = null;
let evilGenerator = null;

const getEvilModule = () => {
  if (!evilModulePromise) {
    evilModulePromise = createEvilCore({
      locateFile: (name) => fileURLToPath(new URL(`../public/wasm/${name}`, import.meta.url)),
      print: () => {},
      printErr: () => {},
    });
  }
  return evilModulePromise;
};

const encodeBoard = (board) => {
  let encoded = 0n;
  flattenBoard(board).slice(0, 16).forEach((value, index) => {
    encoded |= BigInt((Number(value) || 0) & 0xf) << BigInt((15 - index) * 4);
  });
  return encoded;
};

const evilSpawn = async (board, depth) => {
  const module = await getEvilModule();
  const encoded = encodeBoard(board);
  evilGenerator = evilGenerator || new module.EvilGen(encoded);
  evilGenerator.reset_board(encoded);
  const result = evilGenerator.gen_new_num(Math.max(1, Number(depth) || 4));
  const index = Number(result?.[1]);
  const value = Number(result?.[2]);
  if (!Number.isInteger(index) || index < 0 || index >= 16 || !Number.isInteger(value) || value <= 0) {
    throw new Error('evil_spawn_invalid');
  }
  return { index, value };
};

const sameArray = (left, right) => (
  Array.isArray(left)
  && Array.isArray(right)
  && left.length === right.length
  && left.every((value, index) => Number(value) === Number(right[index]))
);

const verifyExpected = (result, expected = {}) => {
  const checks = [
    ['run_id', result.runId, expected.run_id],
    ['seed_hex', result.seedHex, expected.seed_hex],
    ['rules_version', result.rulesVersion, expected.rules_version],
    ['game_id', result.gameId, expected.game_id],
    ['difficulty', result.difficulty, expected.difficulty],
    ['score', result.score, expected.score],
    ['trophy_tier', result.trophyTier, expected.trophy_tier],
    ['highest_tile_exp', result.highestTileExp, expected.highest_tile_exp],
    ['board_rows', result.boardRows, expected.board_rows],
    ['board_cols', result.boardCols, expected.board_cols],
    ['action_count', result.actionCount, expected.action_count],
    ['elapsed_ms', result.elapsedMs, expected.elapsed_ms],
  ];
  for (const [field, actual, claimed] of checks) {
    if (claimed != null && String(actual) !== String(claimed)) throw new Error(`${field}_mismatch`);
  }
  if (expected.final_board != null && !sameArray(result.finalBoard, expected.final_board)) {
    throw new Error('final_board_mismatch');
  }
};

async function verifyRequest(request) {
  const result = await replayMgo1(String(request?.record_encoding || ''), { evilSpawn });
  verifyExpected(result, request?.expected || {});
  return {
    ok: true,
    request_id: request?.request_id ?? null,
    result: {
      run_id: result.runId,
      seed_hex: result.seedHex,
      rules_version: result.rulesVersion,
      game_id: result.gameId,
      difficulty: result.difficulty,
      score: result.score,
      trophy_tier: result.trophyTier,
      highest_tile_exp: result.highestTileExp,
      final_board: result.finalBoard,
      board_rows: result.boardRows,
      board_cols: result.boardCols,
      action_count: result.actionCount,
      elapsed_ms: result.elapsedMs,
    },
  };
}

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of input) {
  if (!line.trim()) continue;
  let request = null;
  try {
    request = JSON.parse(line);
    process.stdout.write(`${JSON.stringify(await verifyRequest(request))}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify({
      ok: false,
      request_id: request?.request_id ?? null,
      error: String(error?.code || error?.message || 'verification_failed').slice(0, 160),
    })}\n`);
  }
}

evilGenerator?.delete?.();
