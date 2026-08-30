import assert from 'node:assert/strict';
import test from 'node:test';

import { BattleController } from '../src/features/battle/engine/battleController.js';
import {
  BattleRouteFormatError,
  decodeTrainerRouteChange,
  parseTrainerBattleRoute,
} from '../src/features/battle/engine/battleRouteCodec.js';
import { scoreBattleStep } from '../src/features/battle/engine/battleScoring.js';
import { encodeBoard } from '../src/features/replay/engine/replayTransition.js';

const RECORD_BYTES = 17;

function change(direction, spawnIndex, spawnValue) {
  const directionCode = ['up', 'down', 'left', 'right'].indexOf(direction);
  return directionCode | ((spawnIndex & 0b1111) << 2) | (spawnValue === 4 ? 0b1000000 : 0);
}

function buildTrainerRoute(initialBoard, steps = []) {
  const buffer = new ArrayBuffer((steps.length + 1) * RECORD_BYTES);
  const view = new DataView(buffer);
  const board = BigInt.asUintN(64, BigInt(initialBoard));
  for (let index = 0; index < 4; index += 1) {
    view.setUint32(1 + index * 4, Number((board >> BigInt(index * 16)) & 0xffffn), true);
  }
  steps.forEach((step, index) => {
    const offset = (index + 1) * RECORD_BYTES;
    view.setUint8(offset, step.change);
    step.rates.forEach((rate, direction) => {
      view.setUint32(offset + 1 + direction * 4, rate, true);
    });
  });
  return buffer;
}

test('parses Trainer 17-byte route records and preserves the uint64 board', () => {
  const initialBoard = 0x0123456789abcdefn;
  const encodedChange = change('right', 14, 4);
  const route = parseTrainerBattleRoute(buildTrainerRoute(initialBoard, [{
    change: encodedChange,
    rates: [1, 2, 3, 4_000_000_000],
  }]));

  assert.equal(route.initialBoard, initialBoard);
  assert.equal(route.moveCount, 1);
  assert.deepEqual(Array.from(route.rates), [1, 2, 3, 4_000_000_000]);
  assert.deepEqual(decodeTrainerRouteChange(encodedChange), {
    directionCode: 3,
    direction: 'right',
    spawnIndex: 14,
    spawnExponent: 2,
    spawnValue: 4,
  });
});

test('rejects malformed Trainer route sizes, header data and rates', () => {
  assert.throws(
    () => parseTrainerBattleRoute(new ArrayBuffer(16)),
    BattleRouteFormatError,
  );
  const invalidHeader = buildTrainerRoute(1n);
  new DataView(invalidHeader).setUint8(0, 1);
  assert.throws(() => parseTrainerBattleRoute(invalidHeader), /invalid_header/u);

  const invalidRate = buildTrainerRoute(1n, [{
    change: change('left', 15, 2),
    rates: [0xffffffff, 0, 0, 0],
  }]);
  assert.throws(() => parseTrainerBattleRoute(invalidRate), /invalid_rate/u);
});

test('battle scoring matches Tester cumulative ratio semantics', () => {
  const scored = scoreBattleStep({
    rates: Uint32Array.from([4_000_000_000, 3_000_000_000, 0, 0]),
    selectedDirection: 'down',
    standardDirection: 'up',
    goodnessOfFit: 0.8,
  });
  assert.equal(scored.ratio, 0.75);
  assert.equal(scored.goodnessDrop, 0.25);
  assert.ok(Math.abs(scored.goodnessOfFit - 0.6) < 1e-12);
  assert.ok(Math.abs(scored.goodnessOfFitDrop - 0.2) < 1e-12);

  const withinTesterTolerance = scoreBattleStep({
    rates: Uint32Array.from([4_000_000_000, 3_999_999_999, 0, 0]),
    selectedDirection: 'down',
    standardDirection: 'up',
  });
  assert.equal(withinTesterTolerance.ratio, 1);
  assert.equal(withinTesterTolerance.perfect, true);
});

test('illegal input does not advance; wrong legal input scores but executes the standard route', () => {
  const initial = encodeBoard([2, 2, 0, 0, ...new Array(12).fill(0)]);
  const route = buildTrainerRoute(initial, [{
    change: change('left', 15, 2),
    rates: [0, 0, 4_000_000_000, 2_000_000_000],
  }]);
  const controller = new BattleController({ route });

  const illegal = controller.input('up');
  assert.equal(illegal.accepted, false);
  assert.equal(illegal.reason, 'illegal_move');
  assert.equal(controller.getState().index, 0);
  assert.equal(controller.getState().goodnessOfFit, 1);

  const accepted = controller.input('right');
  assert.equal(accepted.accepted, true);
  assert.equal(accepted.wrong, true);
  assert.equal(accepted.standardDirection, 'left');
  assert.equal(accepted.scoring.ratio, 0.5);
  assert.deepEqual(accepted.state.board, [4, 0, 0, 0, ...new Array(11).fill(0), 2]);
  assert.deepEqual(accepted.transition.metadata.appear_tile, { index: 15, value: 2 });
  assert.equal(accepted.state.mode, 'complete');
});

test('certaintyStep switches to auto and auto steps do not alter GOF', () => {
  const initial = encodeBoard([2, 2, 0, 0, ...new Array(12).fill(0)]);
  const route = buildTrainerRoute(initial, [
    {
      change: change('left', 15, 2),
      rates: [0, 0, 4_000_000_000, 2_000_000_000],
    },
    {
      change: change('up', 15, 2),
      rates: [4_000_000_000, 1_000_000_000, 0, 0],
    },
  ]);
  const controller = new BattleController({ route, certaintyStep: 1 });
  const first = controller.input('right');
  assert.equal(first.state.mode, 'auto');
  assert.equal(first.state.goodnessOfFit, 0.5);
  assert.equal(controller.input('up').reason, 'auto_mode');

  const automatic = controller.autoStep();
  assert.equal(automatic.accepted, true);
  assert.equal(automatic.auto, true);
  assert.equal(automatic.state.mode, 'complete');
  assert.equal(automatic.state.goodnessOfFit, 0.5);
  assert.deepEqual(automatic.state.board, [4, 0, 0, 2, ...new Array(11).fill(0), 2]);
});

test('variant routes reuse wall and non-merging tile behavior', () => {
  const initial = encodeBoard([2, 2, 32768, 0, 16384, 16384, 0, 0, ...new Array(8).fill(0)]);
  const route = buildTrainerRoute(initial, [{
    change: change('right', 3, 4),
    rates: [0, 0, 0, 4_000_000_000],
  }]);
  const controller = new BattleController({ route, useVariant: true });
  const moved = controller.input('right');
  assert.equal(moved.accepted, true);
  assert.deepEqual(moved.state.board.slice(0, 4), [0, 4, 32768, 4]);
  assert.deepEqual(moved.state.board.slice(4, 8), [0, 0, 16384, 16384]);
});
