import { flattenBoard, POWERUP_KEYS } from './utils.js';

export function defaultPowerupCounts(state) {
  if (state.engine?.legacyName === 'Blitzkrieg') {
    return { bomb: 10, glove: 10, twist: 10 };
  }
  const count = Number(state.difficulty) === 1 ? 1 : 5;
  return { bomb: count, glove: count, twist: count };
}

export function buildPowerupsPayload(state) {
  return {
    enabled: Boolean(state.engine),
    counts: Object.fromEntries(POWERUP_KEYS.map((key) => [key, Number(state.powerupCounts?.[key] || 0)])),
    activeMode: state.activeMode || null,
  };
}

function validBombTargets(engine) {
  return flattenBoard(engine.board).reduce((targets, value, index) => {
    if (value > 0) targets.push(index);
    return targets;
  }, []);
}

function validGloveSources(engine) {
  return validBombTargets(engine);
}

function validGloveTargets(engine) {
  return flattenBoard(engine.board).reduce((targets, value, index) => {
    if (value === 0) targets.push(index);
    return targets;
  }, []);
}

function validTwistTargets(engine) {
  const targets = [];
  for (let row = 0; row < engine.rows - 1; row += 1) {
    for (let col = 0; col < engine.cols - 1; col += 1) {
      const values = [
        engine.board[row][col],
        engine.board[row][col + 1],
        engine.board[row + 1][col],
        engine.board[row + 1][col + 1],
      ];
      if (values.every((value) => value === 0)) continue;
      if (values.some((value) => value === -1)) continue;
      targets.push(row * engine.cols + col);
    }
  }
  return targets;
}

export function buildInteractionPayload(state) {
  const engine = state.engine;
  if (!engine || !state.activeMode) {
    return {
      active: false,
      mode: null,
      targetType: null,
      phase: null,
      validTargets: [],
      selectedIndices: [],
      hintKey: '',
    };
  }

  if (state.activeMode === 'bomb') {
    return {
      active: true,
      mode: 'bomb',
      targetType: 'singleTile',
      phase: 'target',
      validTargets: validBombTargets(engine),
      selectedIndices: [],
      hintKey: 'minigames.powerups.hints.bomb',
    };
  }

  if (state.activeMode === 'glove' && state.interactionPhase === 1) {
    return {
      active: true,
      mode: 'glove',
      targetType: 'singleTile',
      phase: 'source',
      validTargets: validGloveSources(engine),
      selectedIndices: [],
      hintKey: 'minigames.powerups.hints.gloveSource',
    };
  }

  if (state.activeMode === 'glove' && state.interactionPhase === 2) {
    const sourceIndex = Number(state.selectionCache?.sourceIndex ?? -1);
    return {
      active: true,
      mode: 'glove',
      targetType: 'singleTile',
      phase: 'target',
      validTargets: validGloveTargets(engine),
      selectedIndices: sourceIndex >= 0 ? [sourceIndex] : [],
      hintKey: 'minigames.powerups.hints.gloveTarget',
    };
  }

  if (state.activeMode === 'twist') {
    return {
      active: true,
      mode: 'twist',
      targetType: 'subgridTopLeft',
      phase: 'target',
      validTargets: validTwistTargets(engine),
      selectedIndices: [],
      hintKey: 'minigames.powerups.hints.twist',
    };
  }

  return {
    active: false,
    mode: null,
    targetType: null,
    phase: null,
    validTargets: [],
    selectedIndices: [],
    hintKey: '',
  };
}

export function cancelPowerupInteraction(state, { clearAnimation = true } = {}) {
  if (clearAnimation && state.engine) {
    state.engine.clearAnimation();
  }
  state.activeMode = null;
  state.interactionPhase = 0;
  state.selectionCache = null;
}

export function activatePowerup(state, mode) {
  const normalizedMode = String(mode || '').trim().toLowerCase();
  if (!state.engine || !POWERUP_KEYS.includes(normalizedMode)) {
    return false;
  }
  state.engine.clearAnimation();
  if (Number(state.powerupCounts?.[normalizedMode] || 0) <= 0) {
    return false;
  }
  if (state.activeMode === normalizedMode) {
    cancelPowerupInteraction(state);
    return true;
  }
  state.activeMode = normalizedMode;
  state.interactionPhase = 1;
  state.selectionCache = null;
  return true;
}

function consumePowerup(state, mode) {
  state.powerupCounts[mode] = Math.max(0, Number(state.powerupCounts?.[mode] || 0) - 1);
}

function postPowerupUpdate(state) {
  const engine = state.engine;
  if (!engine) return;
  engine.checkGamePassed();
  engine.checkGameOver();
}

function applyBomb(state, index) {
  const engine = state.engine;
  if (!engine) return false;
  const row = Math.floor(index / engine.cols);
  const col = index % engine.cols;
  if (row < 0 || row >= engine.rows || col < 0 || col >= engine.cols) {
    engine.clearAnimation();
    return false;
  }
  if (Number(engine.board[row][col]) <= 0) {
    engine.clearAnimation();
    return false;
  }
  engine.board[row][col] = 0;
  engine.setSpecialEffects([{ type: 'explosion', index: Number(index) }]);
  consumePowerup(state, 'bomb');
  postPowerupUpdate(state);
  return true;
}

function applyGloveStep(state, index) {
  const engine = state.engine;
  if (!engine) return false;
  const row = Math.floor(index / engine.cols);
  const col = index % engine.cols;
  if (row < 0 || row >= engine.rows || col < 0 || col >= engine.cols) {
    cancelPowerupInteraction(state);
    return false;
  }

  if (state.interactionPhase === 1) {
    const value = Number(engine.board[row][col]) || 0;
    if (value <= 0) {
      engine.clearAnimation();
      return false;
    }
    state.interactionPhase = 2;
    state.selectionCache = { sourceIndex: Number(index), value };
    engine.setSpecialEffects([{ type: 'grab', index: Number(index), value }]);
    return false;
  }

  const sourceIndex = Number(state.selectionCache?.sourceIndex ?? -1);
  if (sourceIndex < 0) {
    cancelPowerupInteraction(state);
    return false;
  }
  const sourceRow = Math.floor(sourceIndex / engine.cols);
  const sourceCol = sourceIndex % engine.cols;
  const sourceValue = Number(engine.board[sourceRow][sourceCol]) || 0;
  const success = Number(engine.board[row][col]) === 0 && sourceValue > 0;
  if (success) {
    engine.board[row][col] = sourceValue;
    engine.board[sourceRow][sourceCol] = 0;
    engine.setSpecialEffects([
      {
        type: 'glove_move',
        fromIndex: sourceIndex,
        toIndex: Number(index),
        value: sourceValue,
      },
    ]);
    consumePowerup(state, 'glove');
    postPowerupUpdate(state);
  } else {
    engine.clearAnimation();
  }
  cancelPowerupInteraction(state, { clearAnimation: !success });
  return success;
}

function rotate2x2Clockwise(board, row, col) {
  const topLeft = board[row][col];
  const topRight = board[row][col + 1];
  const bottomLeft = board[row + 1][col];
  const bottomRight = board[row + 1][col + 1];
  board[row][col] = bottomLeft;
  board[row][col + 1] = topLeft;
  board[row + 1][col] = bottomRight;
  board[row + 1][col + 1] = topRight;
}

function applyTwist(state, index) {
  const engine = state.engine;
  if (!engine) return false;
  const row = Math.floor(index / engine.cols);
  const col = index % engine.cols;
  if (row < 0 || col < 0 || row >= engine.rows - 1 || col >= engine.cols - 1) {
    engine.clearAnimation();
    return false;
  }
  const values = [
    engine.board[row][col],
    engine.board[row][col + 1],
    engine.board[row + 1][col],
    engine.board[row + 1][col + 1],
  ];
  if (values.every((value) => value === 0) || values.some((value) => value === -1)) {
    engine.clearAnimation();
    return false;
  }
  const topLeft = row * engine.cols + col;
  const topRight = topLeft + 1;
  const bottomLeft = (row + 1) * engine.cols + col;
  const bottomRight = bottomLeft + 1;
  const mapping = [
    [topLeft, topRight, engine.board[row][col]],
    [topRight, bottomRight, engine.board[row][col + 1]],
    [bottomLeft, topLeft, engine.board[row + 1][col]],
    [bottomRight, bottomLeft, engine.board[row + 1][col + 1]],
  ];
  const twistTiles = mapping
    .filter(([_from, _to, value]) => Number(value) > 0)
    .map(([fromIndex, toIndex, value]) => ({ fromIndex, toIndex, value: Number(value) }));
  rotate2x2Clockwise(engine.board, row, col);
  engine.setSpecialEffects([{ type: 'twist', index: Number(index), tiles: twistTiles }]);
  consumePowerup(state, 'twist');
  postPowerupUpdate(state);
  return true;
}

export function applyTargetAction(state, index) {
  if (!state.engine || !state.activeMode) {
    return false;
  }
  let success = false;
  if (state.activeMode === 'bomb') {
    success = applyBomb(state, Number(index));
    cancelPowerupInteraction(state, { clearAnimation: !success });
    return success;
  }
  if (state.activeMode === 'glove') {
    return applyGloveStep(state, Number(index));
  }
  if (state.activeMode === 'twist') {
    success = applyTwist(state, Number(index));
    cancelPowerupInteraction(state, { clearAnimation: !success });
    return success;
  }
  cancelPowerupInteraction(state);
  return false;
}

export function maybeAwardRandomPowerup(state, scoreDelta) {
  if (Number(scoreDelta) < 300 || Number(scoreDelta) >= 1024) {
    return null;
  }
  const probability = Number(state.difficulty) === 1 ? 0.05 : 0.25;
  if (Math.random() >= probability) {
    return null;
  }
  const awarded = POWERUP_KEYS[Math.floor(Math.random() * POWERUP_KEYS.length)];
  state.powerupCounts[awarded] = Number(state.powerupCounts?.[awarded] || 0) + 1;
  return awarded;
}

export function clonePowerupCounts(counts, fallback) {
  return Object.fromEntries(
    POWERUP_KEYS.map((key) => [key, Math.max(0, Number(counts?.[key] ?? fallback?.[key] ?? 0) || 0)])
  );
}
