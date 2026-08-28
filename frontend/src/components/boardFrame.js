const BOARD_SIZE = 16;

export const cloneBoard = (board) => (
  Array.isArray(board)
    ? board.slice(0, BOARD_SIZE).map((value) => Number(value) || 0)
    : new Array(BOARD_SIZE).fill(0)
);

export const boardsEqual = (left, right) => {
  if (!Array.isArray(left) || !Array.isArray(right)) return false;
  if (left.length < BOARD_SIZE || right.length < BOARD_SIZE) return false;
  for (let index = 0; index < BOARD_SIZE; index += 1) {
    if (Number(left[index]) !== Number(right[index])) return false;
  }
  return true;
};

export const hasBoardAnimation = (metadata) => {
  if (!metadata || typeof metadata !== 'object') return false;
  const appearTile = metadata.appear_tile;
  const hasAppearTile = Boolean(appearTile)
    && Number.isInteger(Number(appearTile.index))
    && Number(appearTile.index) >= 0
    && Number(appearTile.index) < BOARD_SIZE
    && Number(appearTile.value) > 0;
  const {
    direction,
    slide_distances: slideDistances,
    pop_positions: popPositions,
  } = metadata;
  const hasDirectionalAnimation = ['left', 'right', 'up', 'down'].includes(direction)
    && Array.isArray(slideDistances)
    && slideDistances.length === BOARD_SIZE
    && Array.isArray(popPositions)
    && popPositions.length === BOARD_SIZE;
  return hasDirectionalAnimation || hasAppearTile;
};

const cloneAnimationMetadata = (metadata) => {
  if (!metadata || typeof metadata !== 'object') return null;
  const animation = { ...metadata };
  if (Array.isArray(metadata.slide_distances)) {
    animation.slide_distances = Object.freeze([...metadata.slide_distances]);
  }
  if (Array.isArray(metadata.pop_positions)) {
    animation.pop_positions = Object.freeze([...metadata.pop_positions]);
  }
  if (metadata.appear_tile && typeof metadata.appear_tile === 'object') {
    animation.appear_tile = Object.freeze({ ...metadata.appear_tile });
  }
  return Object.freeze(animation);
};

export const createBoardFrame = ({
  revision,
  kind = 'snapshot',
  fromBoard,
  toBoard,
  metadata = null,
} = {}) => {
  const target = cloneBoard(toBoard);
  const source = cloneBoard(fromBoard ?? target);
  const animation = cloneAnimationMetadata(metadata);
  const normalizedKind = kind === 'move' && hasBoardAnimation(animation) ? 'move' : 'snapshot';
  return Object.freeze({
    revision: String(revision ?? '0'),
    kind: normalizedKind,
    fromBoard: Object.freeze(source),
    toBoard: Object.freeze(target),
    metadata: animation,
  });
};

export const createSnapshotBoardFrame = (revision, board) => createBoardFrame({
  revision,
  kind: 'snapshot',
  fromBoard: board,
  toBoard: board,
});

export const createTransitionBoardFrame = (revision, transition, fallbackBoard) => {
  const target = transition?.toBoard ?? fallbackBoard;
  if (!transition || !hasBoardAnimation(transition.metadata)) {
    return createSnapshotBoardFrame(revision, target);
  }
  return createBoardFrame({
    revision,
    kind: 'move',
    fromBoard: transition.fromBoard,
    toBoard: target,
    metadata: transition.metadata,
  });
};

export const boardFrameRenderMode = (settledBoard, frame) => (
  frame?.kind === 'move'
    && hasBoardAnimation(frame.metadata)
    && boardsEqual(settledBoard, frame.fromBoard)
    ? 'animate'
    : 'snapshot'
);
