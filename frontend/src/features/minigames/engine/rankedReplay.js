import { decodeMgo1, canonicalStateDigest } from '../protocol/index.js';
import { MinigameController } from './controller.js';
import {
  CUSTOM_ACTION_KEY,
  CUSTOM_PHASE_KEY,
  verificationState,
} from './rankedRecorder.js';
import { createMinigameRuntime, createVirtualMinigameClock } from './runtime.js';
import { flattenBoard } from './utils.js';

export class MinigameReplayError extends Error {
  constructor(code, message = code) {
    super(message);
    this.name = 'MinigameReplayError';
    this.code = code;
  }
}

const requireAccepted = (controller, code) => {
  if (!controller.lastOperationAccepted) throw new MinigameReplayError(code);
};

export async function replayMgo1(recordEncoding, { evilSpawn = null, requireEnd = true } = {}) {
  const decoded = typeof recordEncoding === 'string' ? decodeMgo1(recordEncoding) : recordEncoding;
  const clock = createVirtualMinigameClock(0);
  const runtime = createMinigameRuntime({
    seedHex: decoded.seedHex,
    clock,
    evilSpawn,
  });
  const controller = new MinigameController({
    difficulty: decoded.difficulty,
    runtime,
  });
  await controller.startGame(decoded.gameId, null, runtime);

  let ended = false;
  let actionCount = 0;
  let elapsedMs = 0;
  for (const action of decoded.actions) {
    clock.advance(action.deltaMs);
    elapsedMs += Math.max(0, Number(action.deltaMs) || 0);
    if (action.type === 'move') {
      await controller.move(action.direction);
      actionCount += 1;
      continue;
    }
    if (action.type === 'bomb') {
      controller.usePowerup('bomb');
      requireAccepted(controller, 'invalid_bomb_activation');
      controller.targetAction(action.index);
      requireAccepted(controller, 'invalid_bomb_target');
      actionCount += 1;
      continue;
    }
    if (action.type === 'glove') {
      controller.usePowerup('glove');
      requireAccepted(controller, 'invalid_glove_activation');
      controller.targetAction(action.source);
      if (controller.interactionPhase !== 2) throw new MinigameReplayError('invalid_glove_source');
      controller.targetAction(action.target);
      requireAccepted(controller, 'invalid_glove_target');
      actionCount += 1;
      continue;
    }
    if (action.type === 'twist') {
      controller.usePowerup('twist');
      requireAccepted(controller, 'invalid_twist_activation');
      controller.targetAction(action.index);
      requireAccepted(controller, 'invalid_twist_target');
      actionCount += 1;
      continue;
    }
    if (action.type === 'custom') {
      const key = CUSTOM_ACTION_KEY[action.actionId];
      const phase = CUSTOM_PHASE_KEY[action.phase];
      if (!key || phase == null) throw new MinigameReplayError('invalid_custom_action');
      controller.triggerCustomAction({ key, phase });
      requireAccepted(controller, 'custom_action_rejected');
      actionCount += 1;
      continue;
    }
    if (action.type === 'tick') {
      if (decoded.gameId !== 'blitzkrieg') throw new MinigameReplayError('unexpected_tick');
      controller.tick();
      requireAccepted(controller, 'premature_tick');
      actionCount += 1;
      continue;
    }
    if (action.type === 'digest') {
      const actual = canonicalStateDigest(verificationState(controller));
      if (actual !== BigInt(action.digest)) throw new MinigameReplayError('digest_mismatch');
      continue;
    }
    if (action.type === 'end') {
      controller.engine.checkGameOver();
      if (!controller.engine.isOver) throw new MinigameReplayError('premature_end');
      ended = true;
      continue;
    }
    throw new MinigameReplayError('unsupported_action');
  }

  if (!ended && requireEnd) throw new MinigameReplayError('missing_end');
  const engine = controller.engine;
  const board = flattenBoard(engine.board).map((value) => Math.trunc(Number(value) || 0));
  return {
    runId: decoded.runId,
    seedHex: decoded.seedHex,
    rulesVersion: decoded.rulesVersion,
    gameId: decoded.gameId,
    difficulty: decoded.difficulty,
    score: Math.max(0, Math.trunc(Number(engine.score) || 0)),
    trophyTier: Math.max(0, Math.min(4, Math.trunc(Number(engine.isPassed) || 0))),
    highestTileExp: Math.max(0, Math.trunc(Number(engine.highestTileExp || engine.maxNum) || 0)),
    finalBoard: board,
    boardRows: Number(engine.rows),
    boardCols: Number(engine.cols),
    actionCount,
    elapsedMs,
    runtimeState: runtime.exportSnapshot(),
  };
}
