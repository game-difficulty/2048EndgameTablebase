import {
  cloneBoard,
  codeToDirection,
  directionToCode,
  emptyPositions,
  randomChoice,
  SPAWN_RATE4,
} from './utils.js';

function splitSegments(line) {
  const segments = [];
  let current = [];
  for (const rawValue of line) {
    const value = Number(rawValue) || 0;
    if (value === -1) {
      if (current.length) {
        segments.push(current);
      }
      segments.push([-1]);
      current = [];
    } else {
      current.push(value);
    }
  }
  if (current.length) {
    segments.push(current);
  }
  return segments;
}

export function mergeLine(line, reverse = false) {
  const source = (reverse ? line.slice().reverse() : line.slice()).map((value) => Number(value) || 0);
  const merged = [];
  let score = 0;
  let skip = false;
  let skipSpecial = false;

  for (const segment of splitSegments(source)) {
    if (segment.length === 1 && segment[0] === -1) {
      merged.push(-1);
      continue;
    }

    const nonZero = segment.filter((value) => value !== 0);
    const nextSegment = [];
    for (let index = 0; index < nonZero.length; index += 1) {
      if (skip) {
        skip = false;
        continue;
      }
      if (skipSpecial) {
        if (nonZero[index] === -3) {
          continue;
        }
        skipSpecial = false;
      }
      if (index + 1 < nonZero.length && nonZero[index] === nonZero[index + 1]) {
        if (nonZero[index] >= 0) {
          const mergedValue = nonZero[index] + 1;
          score += 2 ** mergedValue;
          nextSegment.push(mergedValue);
          skip = true;
        } else if (nonZero[index] === -3) {
          nextSegment.push(-3);
          skipSpecial = true;
        } else {
          nextSegment.push(nonZero[index]);
        }
      } else {
        nextSegment.push(nonZero[index]);
      }
    }
    while (nextSegment.length < segment.length) {
      nextSegment.push(0);
    }
    merged.push(...nextSegment.slice(0, segment.length));
  }

  while (merged.length < source.length) {
    merged.push(0);
  }
  const result = reverse ? merged.slice(0, source.length).reverse() : merged.slice(0, source.length);
  return { line: result, score };
}

export function moveBoard(board, directionOrCode) {
  const direction = typeof directionOrCode === 'number' ? codeToDirection(directionOrCode) : String(directionOrCode || '');
  const source = cloneBoard(board);
  const rows = source.length;
  const cols = rows ? source[0].length : 0;
  const next = source.map((row) => row.slice());
  let score = 0;

  if (direction === 'left' || direction === 'right') {
    for (let row = 0; row < rows; row += 1) {
      const merged = mergeLine(source[row], direction === 'right');
      next[row] = merged.line;
      score += merged.score;
    }
  } else if (direction === 'up' || direction === 'down') {
    for (let col = 0; col < cols; col += 1) {
      const line = source.map((row) => row[col]);
      const merged = mergeLine(line, direction === 'down');
      for (let row = 0; row < rows; row += 1) {
        next[row][col] = merged.line[row];
      }
      score += merged.score;
    }
  }

  return {
    board: next,
    score,
    valid: directionToCode(direction) > 0 && !source.every((row, rowIndex) =>
      row.every((value, colIndex) => value === next[rowIndex][colIndex])
    ),
  };
}

export function genNewNum(board, spawnRate = SPAWN_RATE4) {
  const next = cloneBoard(board);
  const positions = emptyPositions(next);
  if (!positions.length) {
    return { board: next, emptyCount: 0, index: -1, value: -1 };
  }
  const [row, col] = randomChoice(positions);
  const value = Math.random() < Number(spawnRate || 0) ? 2 : 1;
  next[row][col] = value;
  return {
    board: next,
    emptyCount: positions.length,
    index: row * next[0].length + col,
    value,
  };
}

function simulateAnimationLine(line) {
  const merged = [];
  const newLine = [];
  for (const segment of splitSegments(line)) {
    if (segment.length === 1 && segment[0] === -1) {
      newLine.push(-1);
      merged.push(0);
      continue;
    }
    let skip = false;
    let skipSpecial = false;
    const nonZero = segment.filter((value) => value !== 0);
    const tempMerged = [];
    const localMerge = new Array(segment.length).fill(0);
    for (let index = 0; index < nonZero.length; index += 1) {
      if (skip) {
        skip = false;
        continue;
      }
      if (skipSpecial) {
        if (nonZero[index] === -3) {
          continue;
        }
        skipSpecial = false;
      }
      if (index + 1 < nonZero.length && nonZero[index] === nonZero[index + 1]) {
        if (nonZero[index] >= 0) {
          tempMerged.push(nonZero[index] + 1);
          localMerge[tempMerged.length - 1] = 1;
          skip = true;
          continue;
        }
        if (nonZero[index] === -3) {
          tempMerged.push(-3);
          localMerge[tempMerged.length - 1] = 1;
          skipSpecial = true;
          continue;
        }
      }
      tempMerged.push(nonZero[index]);
    }
    while (tempMerged.length < segment.length) {
      tempMerged.push(0);
    }
    newLine.push(...tempMerged.slice(0, segment.length));
    merged.push(...localMerge.slice(0, segment.length));
  }
  while (newLine.length < line.length) {
    newLine.push(0);
    merged.push(0);
  }
  return { newLine: newLine.slice(0, line.length), merged: merged.slice(0, line.length) };
}

function moveDistanceLine(line) {
  let movedDistance = 0;
  let lastTile = 0;
  const moveDistance = new Array(line.length).fill(0);
  for (let index = 0; index < line.length; index += 1) {
    const current = Number(line[index]) || 0;
    if (current === 0) {
      movedDistance += 1;
    } else if (current === -1) {
      movedDistance = 0;
      lastTile = 0;
    } else if (current === -2) {
      lastTile = 0;
    } else if (lastTile === current && current >= 0) {
      moveDistance[index] = movedDistance + 1;
      movedDistance += 1;
      lastTile = 0;
    } else {
      moveDistance[index] = movedDistance;
      lastTile = current;
    }
  }
  return moveDistance;
}

export function computeMoveAnimation(board, direction) {
  const source = cloneBoard(board);
  const rows = source.length;
  const cols = rows ? source[0].length : 0;
  const distances = Array.from({ length: rows }, () => new Array(cols).fill(0));
  const merges = Array.from({ length: rows }, () => new Array(cols).fill(0));
  const isHorizontal = direction === 'left' || direction === 'right';
  const count = isHorizontal ? rows : cols;

  for (let index = 0; index < count; index += 1) {
    const line = isHorizontal ? source[index].slice() : source.map((row) => row[index]);
    const reverse = direction === 'down' || direction === 'right';
    const processLine = reverse ? line.slice().reverse() : line;
    let lineDistances = moveDistanceLine(processLine);
    let lineMerges = simulateAnimationLine(processLine).merged;
    if (reverse) {
      lineDistances = lineDistances.reverse();
      lineMerges = lineMerges.reverse();
    }
    if (isHorizontal) {
      distances[index] = lineDistances;
      merges[index] = lineMerges;
    } else {
      for (let row = 0; row < rows; row += 1) {
        distances[row][index] = lineDistances[row];
        merges[row][index] = lineMerges[row];
      }
    }
  }

  return {
    slide_distances: distances.flat(),
    pop_positions: merges.flat(),
  };
}

export function buildMoveAnimationMetadata(boardBefore, direction, spawnIndex = null, spawnValue = null) {
  const metadata = {
    direction,
    ...computeMoveAnimation(boardBefore, direction),
  };
  if (spawnIndex != null && spawnIndex >= 0 && spawnValue != null && spawnValue > 0) {
    metadata.appearTile = {
      index: Number(spawnIndex),
      value: Number(spawnValue),
    };
  }
  return metadata;
}
