import {
  boardHex,
  buildOptimisticMoveOnlyTransition,
  decodeBoard,
  encodeBoard,
} from '../../../../replay/engine/replayTransition.js';
import {
  decodeTrainerRouteChange,
  parseTrainerBattleRoute,
  routeStepAt,
} from './battleRouteCodec.js';
import { normalizeBattleDirection, scoreBattleStep } from './battleScoring.js';

export class BattleRouteExecutionError extends Error {
  constructor(message) {
    super(message);
    this.name = 'BattleRouteExecutionError';
  }
}

function normalizeCertaintyStep(value, moveCount) {
  if (value == null || value === '') return null;
  const numeric = Number(value);
  if (!Number.isInteger(numeric) || numeric < 0 || numeric > moveCount) {
    throw new RangeError('battle_invalid_certainty_step');
  }
  return numeric;
}

function isParsedRoute(route) {
  return route
    && route.format === 'trainer-route-v1'
    && typeof route.initialBoard === 'bigint'
    && route.changes instanceof Uint8Array
    && route.rates instanceof Uint32Array;
}

function applyStandardStep(boardEncoded, step, useVariant) {
  const board = decodeBoard(boardEncoded);
  const transition = buildOptimisticMoveOnlyTransition(
    board,
    step.direction,
    useVariant,
  );
  if (!transition) throw new BattleRouteExecutionError('battle_route_standard_move_invalid');
  if (transition.board[step.spawnIndex] !== 0) {
    throw new BattleRouteExecutionError('battle_route_spawn_occupied');
  }

  const nextBoard = transition.board.slice();
  nextBoard[step.spawnIndex] = step.spawnValue;
  const nextBoardEncoded = encodeBoard(nextBoard);
  return {
    board,
    nextBoard,
    boardEncoded,
    nextBoardEncoded,
    direction: step.direction,
    spawnIndex: step.spawnIndex,
    spawnValue: step.spawnValue,
    metadata: {
      ...transition.metadata,
      appear_tile: { index: step.spawnIndex, value: step.spawnValue },
    },
  };
}

export class BattleController {
  constructor({
    route,
    certaintyStep = null,
    useVariant = false,
  } = {}) {
    this.route = isParsedRoute(route) ? route : parseTrainerBattleRoute(route);
    this.certaintyStep = normalizeCertaintyStep(certaintyStep, this.route.moveCount);
    this.useVariant = Boolean(useVariant);
    this.index = 0;
    this.boardEncoded = this.route.initialBoard;
    this.goodnessOfFit = 1;
    this.mode = this.#modeForIndex();
  }

  #modeForIndex() {
    if (this.index >= this.route.moveCount) return 'complete';
    if (this.certaintyStep != null && this.index >= this.certaintyStep) return 'auto';
    return 'input';
  }

  #state() {
    return {
      index: this.index,
      moveCount: this.route.moveCount,
      boardEncoded: this.boardEncoded,
      boardHex: boardHex(this.boardEncoded),
      board: decodeBoard(this.boardEncoded),
      goodnessOfFit: this.goodnessOfFit,
      certaintyStep: this.certaintyStep,
      mode: this.mode,
      complete: this.mode === 'complete',
      auto: this.mode === 'auto',
    };
  }

  getState() {
    return this.#state();
  }

  seek(index, { goodnessOfFit = this.goodnessOfFit } = {}) {
    const target = Math.max(0, Math.min(this.route.moveCount, Math.trunc(Number(index) || 0)));
    this.index = 0;
    this.boardEncoded = this.route.initialBoard;
    while (this.index < target) {
      const step = routeStepAt(this.route, this.index);
      const transition = applyStandardStep(this.boardEncoded, step, this.useVariant);
      this.boardEncoded = transition.nextBoardEncoded;
      this.index += 1;
    }
    this.goodnessOfFit = Math.max(0, Math.min(1, Number(goodnessOfFit) || 0));
    this.mode = this.#modeForIndex();
    return this.#state();
  }

  input(direction) {
    const selectedDirection = normalizeBattleDirection(direction);
    if (selectedDirection == null) {
      return { accepted: false, reason: 'invalid_direction', state: this.#state() };
    }
    if (this.mode !== 'input') {
      return {
        accepted: false,
        reason: this.mode === 'auto' ? 'auto_mode' : 'route_complete',
        state: this.#state(),
      };
    }

    const selectedTransition = buildOptimisticMoveOnlyTransition(
      decodeBoard(this.boardEncoded),
      selectedDirection,
      this.useVariant,
    );
    if (!selectedTransition) {
      return { accepted: false, reason: 'illegal_move', state: this.#state() };
    }

    const step = routeStepAt(this.route, this.index);
    const standardTransition = applyStandardStep(this.boardEncoded, step, this.useVariant);
    const scoring = scoreBattleStep({
      rates: step.rates,
      selectedDirection,
      standardDirection: step.direction,
      goodnessOfFit: this.goodnessOfFit,
    });
    const stepIndex = this.index;
    this.index += 1;
    this.boardEncoded = standardTransition.nextBoardEncoded;
    this.goodnessOfFit = scoring.goodnessOfFit;
    this.mode = this.#modeForIndex();

    return {
      accepted: true,
      auto: false,
      wrong: selectedDirection !== step.direction,
      stepIndex,
      selectedDirection,
      standardDirection: step.direction,
      scoring,
      transition: standardTransition,
      state: this.#state(),
    };
  }

  autoStep() {
    if (this.mode !== 'auto') {
      return {
        accepted: false,
        reason: this.mode === 'complete' ? 'route_complete' : 'input_required',
        state: this.#state(),
      };
    }
    const stepIndex = this.index;
    const step = routeStepAt(this.route, stepIndex);
    const transition = applyStandardStep(this.boardEncoded, step, this.useVariant);
    this.index += 1;
    this.boardEncoded = transition.nextBoardEncoded;
    this.mode = this.#modeForIndex();
    return {
      accepted: true,
      auto: true,
      wrong: false,
      stepIndex,
      standardDirection: step.direction,
      transition,
      state: this.#state(),
    };
  }

  runAuto() {
    const transitions = [];
    while (this.mode === 'auto') transitions.push(this.autoStep());
    return transitions;
  }
}

export function createBattleController(options) {
  return new BattleController(options);
}

export { decodeTrainerRouteChange };
