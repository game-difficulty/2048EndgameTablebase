function linePositions(direction) {
  if (direction === "left") {
    return Array.from({ length: 4 }, (_unused, row) =>
      Array.from({ length: 4 }, (__unused, col) => row * 4 + col),
    );
  }
  if (direction === "right") {
    return Array.from({ length: 4 }, (_unused, row) =>
      Array.from({ length: 4 }, (__unused, offset) => row * 4 + (3 - offset)),
    );
  }
  if (direction === "up") {
    return Array.from({ length: 4 }, (_unused, col) =>
      Array.from({ length: 4 }, (__unused, row) => row * 4 + col),
    );
  }
  if (direction === "down") {
    return Array.from({ length: 4 }, (_unused, col) =>
      Array.from({ length: 4 }, (__unused, offset) => (3 - offset) * 4 + col),
    );
  }
  return [];
}

function simulateLine(values) {
  const result = [0, 0, 0, 0];
  const distances = [0, 0, 0, 0];
  const pops = [0, 0, 0, 0];
  let scoreDelta = 0;

  const nonZero = values
    .map((value, index) => [index, Number(value) || 0])
    .filter(([_index, value]) => value !== 0);

  let read = 0;
  let write = 0;
  while (read < nonZero.length) {
    const [sourceIndex, value] = nonZero[read];
    if (read + 1 < nonZero.length && nonZero[read + 1][1] === value) {
      const [nextSourceIndex] = nonZero[read + 1];
      const mergedValue = value * 2;
      result[write] = mergedValue;
      scoreDelta += mergedValue;
      distances[sourceIndex] = sourceIndex - write;
      distances[nextSourceIndex] = nextSourceIndex - write;
      pops[write] = 1;
      read += 2;
    } else {
      result[write] = value;
      distances[sourceIndex] = sourceIndex - write;
      read += 1;
    }
    write += 1;
  }

  return { result, distances, pops, scoreDelta };
}

export function simulateMove(values, direction) {
  const nextBoard = new Array(16).fill(0);
  const slideDistances = new Array(16).fill(0);
  const popPositions = new Array(16).fill(0);
  let scoreDelta = 0;

  for (const positions of linePositions(direction)) {
    const lineValues = positions.map((index) => Number(values[index]) || 0);
    const simulated = simulateLine(lineValues);
    scoreDelta += simulated.scoreDelta;

    positions.forEach((boardIndex, offset) => {
      slideDistances[boardIndex] = simulated.distances[offset];
      if (simulated.pops[offset]) {
        popPositions[boardIndex] = 1;
      }
      nextBoard[boardIndex] = simulated.result[offset];
    });
  }

  return {
    board: nextBoard,
    slideDistances,
    popPositions,
    scoreDelta,
  };
}
