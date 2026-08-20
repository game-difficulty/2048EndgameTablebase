export const REPLAY_DIRECTIONS = Object.freeze(['left', 'right', 'up', 'down']);

export function decodeBoard(encoded) {
  const value = BigInt(encoded || 0n);
  return Array.from({ length: 16 }, (_unused, index) => {
    const exponent = Number((value >> BigInt((15 - index) * 4)) & 0xfn);
    return exponent > 0 ? 2 ** exponent : 0;
  });
}

export function encodeBoard(board) {
  return board.slice(0, 16).reduce((encoded, rawValue, index) => {
    const value = Number(rawValue) || 0;
    const exponent = value > 0 ? Math.min(15, Math.round(Math.log2(value))) : 0;
    return encoded | (BigInt(exponent) << BigInt((15 - index) * 4));
  }, 0n);
}

export function boardHex(encoded) {
  return BigInt(encoded || 0n).toString(16).padStart(16, '0').slice(-16);
}

export function decodeReplayChange(encoded) {
  const value = Number(encoded) || 0;
  return {
    direction: REPLAY_DIRECTIONS[(value >> 5) & 0b11] || null,
    spawnIndex: (value >> 1) & 0b1111,
    spawnExponent: (value & 0b1) + 1,
  };
}

function linePositions(direction) {
  if (direction === 'left') {
    return Array.from({ length: 4 }, (_unused, row) => [0, 1, 2, 3].map((col) => row * 4 + col));
  }
  if (direction === 'right') {
    return Array.from({ length: 4 }, (_unused, row) => [3, 2, 1, 0].map((col) => row * 4 + col));
  }
  if (direction === 'up') {
    return Array.from({ length: 4 }, (_unused, col) => [0, 1, 2, 3].map((row) => row * 4 + col));
  }
  return Array.from({ length: 4 }, (_unused, col) => [3, 2, 1, 0].map((row) => row * 4 + col));
}

function mergeOrientedLine(values, { barriers, nonMerging }) {
  const output = new Array(values.length).fill(0);
  let index = 0;
  while (index < values.length) {
    if (barriers.has(values[index])) {
      output[index] = values[index];
      index += 1;
      continue;
    }
    let nextBarrier = index;
    while (nextBarrier < values.length && !barriers.has(values[nextBarrier])) nextBarrier += 1;
    const nonZero = values.slice(index, nextBarrier).filter((value) => value !== 0);
    let read = 0;
    let write = index;
    while (read < nonZero.length) {
      if (
        read + 1 < nonZero.length
        && nonZero[read] === nonZero[read + 1]
        && !nonMerging.has(nonZero[read])
      ) {
        output[write] = nonZero[read] * 2;
        read += 2;
      } else {
        output[write] = nonZero[read];
        read += 1;
      }
      write += 1;
    }
    index = nextBarrier;
  }
  return output;
}

function moveBoard(board, direction, useVariant) {
  const output = board.slice();
  const options = useVariant
    ? { barriers: new Set([32768]), nonMerging: new Set([32768, 16384]) }
    : { barriers: new Set(), nonMerging: new Set([32768]) };
  for (const positions of linePositions(direction)) {
    const merged = mergeOrientedLine(positions.map((index) => board[index]), options);
    positions.forEach((boardIndex, offset) => {
      output[boardIndex] = merged[offset];
    });
  }
  return output;
}

function hasOnlyZerosBetween(board, first, second, horizontal) {
  if (horizontal) {
    const row = Math.floor(first / 4);
    for (let col = Math.min(first % 4, second % 4) + 1; col < Math.max(first % 4, second % 4); col += 1) {
      if (board[row * 4 + col] !== 0) return false;
    }
    return true;
  }
  const col = first % 4;
  for (let row = Math.min(Math.floor(first / 4), Math.floor(second / 4)) + 1; row < Math.max(Math.floor(first / 4), Math.floor(second / 4)); row += 1) {
    if (board[row * 4 + col] !== 0) return false;
  }
  return true;
}

function prepareClassic32kMerge(board, direction) {
  const positions = board.reduce((items, value, index) => {
    if (value === 32768) items.push(index);
    return items;
  }, []);
  if (positions.length !== 2) return board.slice();
  const [first, second] = positions;
  const sameRow = Math.floor(first / 4) === Math.floor(second / 4);
  const sameColumn = first % 4 === second % 4;
  const horizontal = sameRow && (direction === 'left' || direction === 'right');
  const vertical = sameColumn && (direction === 'up' || direction === 'down');
  if (!(horizontal || vertical) || !hasOnlyZerosBetween(board, first, second, horizontal)) return board.slice();
  const adjusted = board.slice();
  adjusted[first] = 16384;
  adjusted[second] = 16384;
  return adjusted;
}

function moveDistanceLine(line, nonMerging) {
  let movedDistance = 0;
  let lastTile = 0;
  const distances = new Array(line.length).fill(0);
  line.forEach((value, index) => {
    if (value === 0) {
      movedDistance += 1;
    } else if (value === -1) {
      movedDistance = 0;
      lastTile = 0;
    } else if (value === -2) {
      lastTile = 0;
    } else if (lastTile === value && !nonMerging.has(value)) {
      distances[index] = movedDistance + 1;
      movedDistance += 1;
      lastTile = 0;
    } else {
      distances[index] = movedDistance;
      lastTile = value;
    }
  });
  return distances;
}

function animationMetadata(board, direction, useVariant, spawnIndex, spawnValue) {
  const normalized = board.map((value) => (useVariant && value === 32768 ? -1 : value));
  const nonMerging = new Set(useVariant ? [32768, 16384] : [32768]);
  const barriers = new Set([-1]);
  const slide = new Array(16).fill(0);
  const pops = new Array(16).fill(0);
  for (const positions of linePositions(direction)) {
    const values = positions.map((index) => normalized[index]);
    const distances = moveDistanceLine(values, nonMerging);
    positions.forEach((boardIndex, offset) => {
      slide[boardIndex] = distances[offset];
    });

    let segmentStart = 0;
    while (segmentStart < values.length) {
      if (barriers.has(values[segmentStart])) {
        segmentStart += 1;
        continue;
      }
      let segmentEnd = segmentStart;
      while (segmentEnd < values.length && !barriers.has(values[segmentEnd])) segmentEnd += 1;
      const nonZero = values.slice(segmentStart, segmentEnd).filter((value) => value !== 0);
      let read = 0;
      let write = segmentStart;
      while (read < nonZero.length) {
        if (
          read + 1 < nonZero.length
          && nonZero[read] === nonZero[read + 1]
          && !nonMerging.has(nonZero[read])
        ) {
          pops[positions[write]] = 1;
          read += 2;
        } else {
          read += 1;
        }
        write += 1;
      }
      segmentStart = segmentEnd;
    }
  }
  return {
    direction,
    slide_distances: slide,
    pop_positions: pops,
    appear_tile: { index: spawnIndex, value: spawnValue },
  };
}

export function buildOptimisticMoveOnlyTransition(
  board,
  direction,
  useVariant = false,
) {
  if (!Array.isArray(board) || board.length < 16 || !REPLAY_DIRECTIONS.includes(direction)) {
    return null;
  }
  const originalBoard = board.slice(0, 16).map((value) => Number(value) || 0);
  const boardForMove = useVariant
    ? originalBoard.slice()
    : prepareClassic32kMerge(originalBoard, direction);
  const movedBoard = moveBoard(boardForMove, direction, useVariant);
  if (encodeBoard(movedBoard) === encodeBoard(boardForMove)) return null;
  const encoded = encodeBoard(movedBoard);
  const metadata = animationMetadata(
    useVariant ? originalBoard : boardForMove,
    direction,
    useVariant,
    -1,
    0,
  );
  delete metadata.appear_tile;
  return {
    board: movedBoard,
    boardEncoded: encoded,
    hex: boardHex(encoded),
    spawnIndex: -1,
    spawnValue: 0,
    metadata,
  };
}

export function buildOptimisticMoveTransition(
  board,
  direction,
  useVariant = false,
  spawnRate4 = 0.1,
  randomSource = Math.random,
) {
  const moveOnly = buildOptimisticMoveOnlyTransition(board, direction, useVariant);
  if (!moveOnly) return null;
  const originalBoard = board.slice(0, 16).map((value) => Number(value) || 0);
  const boardForMove = useVariant
    ? originalBoard.slice()
    : prepareClassic32kMerge(originalBoard, direction);
  const movedBoard = moveOnly.board;

  const emptyIndices = movedBoard.reduce((indices, value, index) => {
    if (value === 0) indices.push(index);
    return indices;
  }, []);
  if (!emptyIndices.length) return null;

  const chooseRandom = typeof randomSource === 'function' ? randomSource : Math.random;
  const positionRoll = Math.max(0, Math.min(0.999999999999, Number(chooseRandom()) || 0));
  const spawnIndex = emptyIndices[Math.floor(positionRoll * emptyIndices.length)];
  const normalizedRate4 = Math.max(0, Math.min(1, Number(spawnRate4) || 0));
  const spawnValue = (Number(chooseRandom()) || 0) < normalizedRate4 ? 4 : 2;
  const nextBoard = movedBoard.slice();
  nextBoard[spawnIndex] = spawnValue;

  return {
    board: nextBoard,
    boardEncoded: encodeBoard(nextBoard),
    hex: boardHex(encodeBoard(nextBoard)),
    spawnIndex,
    spawnValue,
    metadata: animationMetadata(
      useVariant ? originalBoard : boardForMove,
      direction,
      useVariant,
      spawnIndex,
      spawnValue,
    ),
  };
}

export function buildStepTransition(replay, step, useVariant = false) {
  if (!replay || step < 0 || step >= replay.moveCount) return null;
  const boardEncoded = replay.boards[step];
  const change = decodeReplayChange(replay.changes[step]);
  if (!change.direction) return null;
  const originalBoard = decodeBoard(boardEncoded);
  const boardForMove = useVariant
    ? originalBoard.slice()
    : prepareClassic32kMerge(originalBoard, change.direction);
  const movedBoard = moveBoard(boardForMove, change.direction, useVariant);
  if (encodeBoard(movedBoard) === encodeBoard(boardForMove)) return null;
  if (movedBoard[change.spawnIndex] !== 0) return null;
  const spawnValue = 2 ** change.spawnExponent;
  const nextBoard = movedBoard.slice();
  nextBoard[change.spawnIndex] = spawnValue;
  return {
    boardEncoded,
    nextBoardEncoded: encodeBoard(nextBoard),
    metadata: animationMetadata(
      useVariant ? originalBoard : boardForMove,
      change.direction,
      useVariant,
      change.spawnIndex,
      spawnValue,
    ),
  };
}

export function transitionMatchesNextSnapshot(replay, step, useVariant = false) {
  const transition = buildStepTransition(replay, step, useVariant);
  if (!transition) return false;
  if (step + 1 < replay.moveCount) return transition.nextBoardEncoded === replay.boards[step + 1];
  return replay.terminalBoard == null || transition.nextBoardEncoded === replay.terminalBoard;
}

export function boardForReplayStep(replay, step, useVariant = false) {
  if (!replay?.moveCount) return 0n;
  if (step <= 0) return replay.boards[0];
  if (step < replay.moveCount) return replay.boards[step];
  if (replay.terminalBoard != null) return replay.terminalBoard;
  return buildStepTransition(replay, replay.moveCount - 1, useVariant)?.nextBoardEncoded
    ?? replay.boards[replay.moveCount - 1];
}
