import { decodeBoard } from '../../../../replay/engine/replayTransition.js';
import { applyStandardStep } from './battleController.js';
import { routeStepAt } from './battleRouteCodec.js';
import { correctionOverlayForResult } from '../../../core/battleCorrection.js';

export function createRoutePresentation(controller) {
  const { route, useVariant } = controller;
  const boards = [route.initialBoard];
  const boardAt = (index) => {
    while (boards.length <= index) {
      const i = boards.length - 1;
      boards.push(applyStandardStep(boards[i], routeStepAt(route, i), useVariant).nextBoardEncoded);
    }
    return boards[index];
  };
  return (result, now) => {
    const overlay = correctionOverlayForResult(result, now);
    let index = overlay?.previousRouteIndex ?? Number(result.route_index || 0);
    let nextAt = overlay?.visibleUntil ?? Infinity;
    const playback = result.mode_data?.auto_playback;
    if (!overlay && playback && result.status === 'completed') {
      const started = Date.parse(playback.started_at);
      const interval = Math.max(50, Number(playback.step_ms) || 150);
      if (Number.isFinite(started)) {
        const elapsed = Math.max(0, Math.floor((now - started) / interval));
        index = Math.min(route.moveCount, Number(playback.from_index) + elapsed);
        if (index < route.moveCount) nextAt = started + (elapsed + 1) * interval;
      }
    }
    index = Math.max(0, Math.min(route.moveCount, Math.trunc(index)));
    const board = decodeBoard(boardAt(index));
    const step = index > 0
      ? applyStandardStep(boardAt(index - 1), routeStepAt(route, index - 1), useVariant)
      : null;
    return { index, board, overlay, nextAt, transition: step && {
      fromBoard: step.board, toBoard: step.nextBoard, metadata: step.metadata,
    } };
  };
}
