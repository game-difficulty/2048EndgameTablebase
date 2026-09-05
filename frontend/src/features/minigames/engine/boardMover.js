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

  for (const segment of splitSegments(source)) {
    if (segment.length === 1 && segment[0] === -1) {
      merged.push(-1);
      continue;
    }

    const nonZero = segment.filter((value) => value !== 0);
    const nextSegment = [];
    let skip = false;
    let skipSpecial = false;
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

export function genNewNum(board, spawnRate = SPAWN_RATE4, rng = null) {
  const next = cloneBoard(board);
  const positions = emptyPositions(next);
  if (!positions.length) {
    return { board: next, emptyCount: 0, index: -1, value: -1 };
  }
  const [row, col] = randomChoice(positions, rng);
  const roll = rng && typeof rng.nextFloat === 'function' ? rng.nextFloat() : Math.random();
  const value = roll < Number(spawnRate || 0) ? 2 : 1;
  next[row][col] = value;
  return {
    board: next,
    emptyCount: positions.length,
    index: row * next[0].length + col,
    value,
  };
}

function traceLineAnimation(line) {
  const values = line.map((value) => Number(value) || 0);
  const nextLine = [];
  const distances = new Array(values.length).fill(0);
  const pops = new Array(values.length).fill(0);
  let segmentStart = 0;

  while (segmentStart < values.length) {
    if (values[segmentStart] === -1) {
      nextLine.push(-1);
      segmentStart += 1;
      continue;
    }

    let segmentEnd = segmentStart;
    while (segmentEnd < values.length && values[segmentEnd] !== -1) {
      segmentEnd += 1;
    }

    const sources = [];
    for (let index = segmentStart; index < segmentEnd; index += 1) {
      if (values[index] !== 0) sources.push([index, values[index]]);
    }

    let target = segmentStart;
    let sourceIndex = 0;
    while (sourceIndex < sources.length) {
      const value = sources[sourceIndex][1];
      let groupEnd = sourceIndex + 1;
      let resultValue = value;

      if (value === -3) {
        while (groupEnd < sources.length && sources[groupEnd][1] === -3) groupEnd += 1;
      } else if (value >= 0 && groupEnd < sources.length && sources[groupEnd][1] === value) {
        groupEnd += 1;
        resultValue = value + 1;
      }

      for (let index = sourceIndex; index < groupEnd; index += 1) {
        distances[sources[index][0]] = sources[index][0] - target;
      }
      if (groupEnd - sourceIndex > 1) pops[target] = 1;

      nextLine.push(resultValue);
      target += 1;
      sourceIndex = groupEnd;
    }

    while (target < segmentEnd) {
      nextLine.push(0);
      target += 1;
    }
    segmentStart = segmentEnd;
  }

  return { nextLine, distances, pops };
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
    const traced = traceLineAnimation(processLine);
    let lineDistances = traced.distances;
    let lineMerges = traced.pops;
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
  if (spawnIndex != null && spawnIndex >= 0 && spawnValue != null) {
    metadata.appearTile = {
      index: Number(spawnIndex),
      value: Number(spawnValue),
    };
  }
  return metadata;
}
