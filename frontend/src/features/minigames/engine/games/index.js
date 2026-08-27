import { buildMoveAnimationMetadata, genNewNum, moveBoard } from '../boardMover.js';
import { BaseMinigameEngine } from '../baseEngine.js';
import { generateEvilSpawn } from '../evilGenAdapter.js';
import {
  boardsEqual,
  cloneBoard,
  countCells,
  createBoard,
  emptyPositions,
  flattenBoard,
  formatCompactTile,
  positiveMax,
  randomChoice,
  randomSample,
  SPAWN_RATE4,
} from '../utils.js';

const DESIGN_MASTER_PATTERNS = {
  'Design Master1': [
    [0, 0, 0, 0],
    [0, 2, 1, 0],
    [0, 1, 2, 0],
    [0, 0, 0, 0],
  ],
  'Design Master2': [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ],
  'Design Master3': [
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
  ],
  'Design Master4': [
    [0, 0, 0, 0],
    [0, 0.0078125, 1, 0.5],
    [0, 0.015625, 2, 0.25],
    [0, 0.03125, 0.0625, 0.125],
  ],
};

function countZeros(values) {
  return values.reduce((count, value) => count + (Number(value) === 0 ? 1 : 0), 0);
}

function findMergePositions(board, direction) {
  const rows = board.length;
  const cols = rows ? board[0].length : 0;
  const result = Array.from({ length: rows }, () => new Array(cols).fill(0));
  const horizontal = direction === 'left' || direction === 'right';
  const count = horizontal ? rows : cols;
  for (let index = 0; index < count; index += 1) {
    const line = horizontal ? board[index].slice() : board.map((row) => row[index]);
    const reverse = direction === 'right' || direction === 'down';
    const processed = reverse ? line.slice().reverse() : line;
    const merged = [];
    const nonZero = processed.filter((value) => value !== 0 && value !== -1);
    let write = 0;
    let read = 0;
    while (read < nonZero.length) {
      if (read + 1 < nonZero.length && nonZero[read] === nonZero[read + 1]) {
        merged[write] = 1;
        read += 2;
      } else {
        merged[write] = 0;
        read += 1;
      }
      write += 1;
    }
    while (merged.length < line.length) merged.push(0);
    const finalMerged = reverse ? merged.slice(0, line.length).reverse() : merged.slice(0, line.length);
    if (horizontal) {
      result[index] = finalMerged;
    } else {
      for (let row = 0; row < rows; row += 1) result[row][index] = finalMerged[row];
    }
  }
  return result;
}

function weightedExponentChoice(rng = null) {
  const exponents = [2, 3, 4, 5, 6, 7, 8, 9, 10];
  const weights = exponents.map((exponent) => 1 / (exponent ** 1.5));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  let roll = (rng?.nextFloat?.() ?? Math.random()) * total;
  for (let index = 0; index < exponents.length; index += 1) {
    roll -= weights[index];
    if (roll <= 0) return exponents[index];
  }
  return exponents[exponents.length - 1];
}

export class DesignMasterEngine extends BaseMinigameEngine {
  constructor(definition, difficulty, snapshot = null, runtime = null) {
    super(definition, difficulty, snapshot, { deferSetup: true, runtime });
    this.pattern = DESIGN_MASTER_PATTERNS[definition.legacyName].map((row) => row.slice());
    this.initialize(snapshot);
    this.currentMaxNum = Math.max(Number(this.maxNum) || 0, 7);
  }

  setupNewGame() {
    super.setupNewGame();
    this.currentMaxNum = 7;
  }

  buildTargetBoard() {
    const scale = Math.max(this.currentMaxNum, this.maxNum) + 1;
    return this.pattern.map((row) => row.map((value) => Number(value) * (2 ** scale)));
  }

  formattedTargetLines() {
    return this.buildTargetBoard().map((row) =>
      row.map((value) => (value ? formatCompactTile(value).padStart(3, ' ') : '  _')).join(' ')
    );
  }

  buildViewState() {
    const view = super.buildViewState();
    view.smallLabels = this.buildTargetBoard().flat().map((value) => formatCompactTile(value));
    return view;
  }

  buildHud() {
    const hud = super.buildHud();
    hud.customPanels = [{ type: 'patternText', title: 'Target Pattern', lines: this.formattedTargetLines() }];
    return hud;
  }

  getInfoText() {
    return `Fit a particular pattern.\n${this.formattedTargetLines().join('\n')}`;
  }

  checkPattern() {
    for (let num = 8; num < 15; num += 1) {
      let ok = true;
      for (let row = 0; row < this.rows; row += 1) {
        for (let col = 0; col < this.cols; col += 1) {
          const patternValue = Number(this.pattern[row]?.[col] || 0);
          if (!patternValue) continue;
          const boardValue = this.board[row][col] > 0 ? 2 ** this.board[row][col] : 0;
          if (Math.abs(boardValue - patternValue * (2 ** num)) > 0.001) {
            ok = false;
          }
        }
      }
      if (ok) return num;
    }
    return false;
  }

  checkGamePassed() {
    this.refreshHighestTileExp();
    const baseline = Math.max(Number(this.maxNum) || 0, 7);
    const patternLevel = this.checkPattern();
    if (!patternLevel || patternLevel <= baseline) {
      this.currentMaxNum = baseline;
      return;
    }
    this.currentMaxNum = patternLevel;
    const level = { 10: 'gold', 9: 'silver', 8: 'bronze' }[this.currentMaxNum] || 'gold';
    if (this.currentMaxNum > this.maxNum) {
      this.maxNum = this.currentMaxNum;
      this.isPassed = { 10: 3, 9: 2, 8: 1 }[this.maxNum] || 4;
      this.queueMessage('trophy', {
        level,
        message: `You achieved ${2 ** this.maxNum}! You get a ${level} trophy!`,
      });
    } else {
      this.queueMessage('trophy', {
        level,
        message: `You achieved ${2 ** this.currentMaxNum}! Take it further!`,
      });
    }
  }
}

export class GravityTwistEngine extends BaseMinigameEngine {
  constructor(definition, difficulty, snapshot = null, runtime = null) {
    super(definition, difficulty, snapshot, { deferSetup: true, runtime });
    this.variant = definition.legacyName.endsWith('2') ? 2 : 1;
    this.initialize(snapshot);
  }

  compressDown() {
    const board = cloneBoard(this.board);
    for (let col = 0; col < this.cols; col += 1) {
      const nonZero = board.map((row) => row[col]).filter((value) => value !== 0);
      const zeros = new Array(this.rows - nonZero.length).fill(0);
      const nextCol = zeros.concat(nonZero);
      for (let row = 0; row < this.rows; row += 1) {
        board[row][col] = nextCol[row];
      }
    }
    return board;
  }

  buildDropOnlyFollowUp(before, after) {
    const slideDistances = new Array(this.rows * this.cols).fill(0);
    const popPositions = new Array(this.rows * this.cols).fill(0);
    for (let col = 0; col < this.cols; col += 1) {
      const sourceRows = [];
      const targetRows = [];
      for (let row = 0; row < this.rows; row += 1) {
        if (before[row][col] > 0) sourceRows.push(row);
        if (after[row][col] > 0) targetRows.push(row);
      }
      sourceRows.reverse().forEach((sourceRow, index) => {
        const targetRow = targetRows.slice().reverse()[index];
        slideDistances[sourceRow * this.cols + col] = Math.max(0, targetRow - sourceRow);
      });
    }
    return {
      kind: 'move',
      direction: 'down',
      slide_distances: slideDistances,
      pop_positions: popPositions,
      delayMs: 360,
      durationMs: 180,
      lockInput: false,
    };
  }

  afterGenNum() {
    const before = cloneBoard(this.board);
    if (this.variant === 1) {
      this.board = this.compressDown();
    } else {
      const result = moveBoard(this.board, 'down');
      if (result.valid) {
        this.board = result.board;
        this.score += result.score;
        this.maxScore = Math.max(this.maxScore, this.score);
      }
    }
    if (!boardsEqual(this.board, before)) {
      if (this.variant === 1) {
        this.setFollowUpAnimation(this.buildDropOnlyFollowUp(before, this.board));
      } else {
        const metadata = buildMoveAnimationMetadata(before, 'down');
        this.setFollowUpAnimation({
          kind: 'move',
          direction: 'down',
          slide_distances: metadata.slide_distances,
          pop_positions: metadata.pop_positions,
          delayMs: 360,
          durationMs: 180,
          lockInput: false,
        });
      }
    }
  }

  getInfoText() {
    return 'The tiles are affected by the unusual gravity.';
  }
}

export class ColumnChaosEngine extends BaseMinigameEngine {
  constructor(definition, difficulty, snapshot = null, runtime = null) {
    super(definition, difficulty, snapshot, { deferSetup: true, runtime });
    this.countDown = 40 - 10 * this.difficulty;
    this.initialize(snapshot);
  }

  loadLegacyExtra(extra) {
    this.countDown = Number(extra?.[0] ?? 40 - 10 * this.difficulty) || 0;
  }

  exportLegacyExtra() {
    return [this.countDown];
  }

  setupNewGame() {
    super.setupNewGame();
    this.countDown = 40 - 10 * this.difficulty;
  }

  afterGenNum() {
    this.countDown -= 1;
    if (this.countDown > 0) return;
    this.countDown = 40 - 10 * this.difficulty;
    const [col1, col2] = randomSample(Array.from({ length: this.cols }, (_value, index) => index), 2, this.runtime.rng);
    const effects = [];
    for (let row = 0; row < this.rows; row += 1) {
      const leftValue = this.board[row][col1];
      const rightValue = this.board[row][col2];
      if (leftValue > 0) effects.push({ type: 'column_swap_move', fromIndex: row * this.cols + col1, toIndex: row * this.cols + col2, value: leftValue, durationMs: 1250 });
      if (rightValue > 0) effects.push({ type: 'column_swap_move', fromIndex: row * this.cols + col2, toIndex: row * this.cols + col1, value: rightValue, durationMs: 1250 });
      this.board[row][col1] = rightValue;
      this.board[row][col2] = leftValue;
    }
    if (effects.length) {
      this.setFollowUpAnimation({ kind: 'effects', delayMs: 270, durationMs: 1250, effects });
    }
  }

  buildHud() {
    const hud = super.buildHud();
    hud.customPanels = [{ type: 'remainingSteps', title: 'Next Chaos', value: this.countDown, suffix: 'steps' }];
    return hud;
  }

  getInfoText() {
    return 'Unpredictable shifts in columns are coming!';
  }
}

export class FerrisWheelEngine extends ColumnChaosEngine {
  constructor(definition, difficulty, snapshot = null, runtime = null) {
    super(definition, difficulty, snapshot, runtime);
    this.countDown = Number(this.countDown) || 40 - 10 * this.difficulty;
  }

  static OUTER_RING = [
    [0, 0], [0, 1], [0, 2], [0, 3],
    [1, 3], [2, 3],
    [3, 3], [3, 2], [3, 1], [3, 0],
    [2, 0], [1, 0],
  ];

  setupNewGame() {
    BaseMinigameEngine.prototype.setupNewGame.call(this);
    this.countDown = 40 - 10 * this.difficulty;
  }

  afterGenNum() {
    this.countDown -= 1;
    if (this.countDown > 0) return;
    this.countDown = 40 - 10 * this.difficulty;
    const values = FerrisWheelEngine.OUTER_RING.map(([row, col]) => this.board[row][col]);
    const rotated = values.slice(-1).concat(values.slice(0, -1));
    const effects = [];
    FerrisWheelEngine.OUTER_RING.forEach(([fromRow, fromCol], index) => {
      const [toRow, toCol] = FerrisWheelEngine.OUTER_RING[(index + 1) % FerrisWheelEngine.OUTER_RING.length];
      const value = values[index];
      if (value > 0) effects.push({ type: 'ring_rotate_move', fromIndex: fromRow * this.cols + fromCol, toIndex: toRow * this.cols + toCol, value, durationMs: 1000 });
    });
    FerrisWheelEngine.OUTER_RING.forEach(([row, col], index) => {
      this.board[row][col] = rotated[index];
    });
    if (effects.length) {
      this.setFollowUpAnimation({ kind: 'effects', delayMs: 270, durationMs: 1000, effects });
    }
  }

  buildHud() {
    const hud = super.buildHud();
    hud.customPanels = [{ type: 'remainingSteps', title: 'Next Rotation', value: this.countDown, suffix: 'steps' }];
    return hud;
  }

  getInfoText() {
    return 'The Earth revolves around the Sun.';
  }
}

export class MysteryMergeEngine extends BaseMinigameEngine {
  constructor(definition, difficulty, snapshot = null, runtime = null) {
    super(definition, difficulty, snapshot, { deferSetup: true, runtime });
    this.variant = definition.legacyName.endsWith('2') ? 2 : 1;
    this.peekCount = 0;
    this.peekActive = false;
    this.revealAll = false;
    this.masked = createBoard(4, 4, false);
    this.initialize(snapshot);
  }

  loadLegacyExtra(extra) {
    this.peekCount = Number(extra?.[0] || 0);
    this.masked = Array.isArray(extra?.[1]?.[0])
      ? extra[1].map((row) => row.map(Boolean))
      : createBoard(this.rows, this.cols, false);
    this.revealAll = false;
    this.peekActive = false;
  }

  exportLegacyExtra() {
    return [this.peekCount, this.masked.map((row) => row.map(Boolean))];
  }

  setupNewGame() {
    this.masked = createBoard(this.rows, this.cols, false);
    this.peekCount = 0;
    this.peekActive = false;
    this.revealAll = false;
    super.setupNewGame();
    if (this.variant === 2 && this.newtilePos >= 0) {
      this.afterGenNum();
    }
  }

  canPeek() {
    if (this.difficulty === 0) return true;
    return this.peekCount <= Math.floor(this.score / 10000);
  }

  remainingPeekCount() {
    if (this.difficulty === 0) return '\u221e';
    return String(Math.max(0, Math.floor(this.score / 10000) - this.peekCount + 1));
  }

  handleCustomAction(key, phase = 'trigger') {
    if (String(key || '').toLowerCase() !== 'peek') return false;
    const normalizedPhase = String(phase || 'trigger').toLowerCase();
    if (normalizedPhase === 'start') {
      if (this.hasPossibleMove() && this.canPeek()) {
        this.peekActive = true;
        this.peekCount += 1;
        return true;
      }
      return false;
    }
    if (normalizedPhase === 'end' || normalizedPhase === 'cancel') {
      if (this.peekActive) {
        this.peekActive = false;
        return true;
      }
      return false;
    }
    if (this.hasPossibleMove() && this.canPeek()) {
      this.peekActive = !this.peekActive;
      if (this.peekActive) this.peekCount += 1;
      return true;
    }
    return false;
  }

  updateMaskLine(line, mask, reverse = false) {
    const currentLine = reverse ? line.slice().reverse() : line.slice();
    const currentMask = reverse ? mask.slice().reverse() : mask.slice();
    const nonZero = [];
    const nonZeroMask = [];
    currentLine.forEach((value, index) => {
      if (value !== 0) {
        nonZero.push(value);
        nonZeroMask.push(Boolean(currentMask[index]));
      }
    });
    const mergedMask = [];
    let skip = false;
    for (let index = 0; index < nonZero.length; index += 1) {
      if (skip) {
        skip = false;
        continue;
      }
      if (index + 1 < nonZero.length && nonZero[index] === nonZero[index + 1]) {
        mergedMask.push(false);
        skip = true;
      } else {
        mergedMask.push(Boolean(nonZeroMask[index]));
      }
    }
    while (mergedMask.length < currentMask.length) mergedMask.push(false);
    return reverse ? mergedMask.slice(0, currentMask.length).reverse() : mergedMask.slice(0, currentMask.length);
  }

  beforeMove(direct) {
    const direction = ['left', 'right', 'up', 'down'][direct - 1];
    if (this.variant === 1) {
      this.masked = findMergePositions(this.board, direction).map((row) => row.map(Boolean));
    } else if (direct === 1 || direct === 2) {
      for (let row = 0; row < this.rows; row += 1) {
        this.masked[row] = this.updateMaskLine(this.board[row], this.masked[row], direct === 2);
      }
    } else {
      for (let col = 0; col < this.cols; col += 1) {
        const line = this.board.map((row) => row[col]);
        const mask = this.masked.map((row) => row[col]);
        const updated = this.updateMaskLine(line, mask, direct === 4);
        for (let row = 0; row < this.rows; row += 1) this.masked[row][col] = updated[row];
      }
    }
    this.peekActive = false;
    this.revealAll = false;
  }

  afterGenNum() {
    if (this.variant === 2 && this.newtilePos >= 0) {
      const row = Math.floor(this.newtilePos / this.cols);
      const col = this.newtilePos % this.cols;
      this.masked[row][col] = true;
    }
  }

  checkGameOver() {
    if (this.hasPossibleMove()) return;
    this.revealAll = true;
    this.peekActive = false;
    this.isOver = true;
  }

  buildViewState() {
    const view = super.buildViewState();
    const hidden = createBoard(this.rows, this.cols, false);
    if (!this.peekActive && !this.revealAll) {
      for (let row = 0; row < this.rows; row += 1) {
        for (let col = 0; col < this.cols; col += 1) {
          const index = row * this.cols + col;
          const value = this.board[row][col];
          if (value <= 0) continue;
          hidden[row][col] = this.variant === 1
            ? index !== this.newtilePos && !this.masked[row][col]
            : Boolean(this.masked[row][col]) || index === this.newtilePos;
        }
      }
    }
    view.hiddenMask = hidden.flat();
    return view;
  }

  buildHud() {
    const hud = super.buildHud();
    hud.customPanels = [{
      type: 'actionButton',
      title: 'Peek',
      key: 'peek',
      label: 'Peek',
      hold: true,
      pressed: Boolean(this.peekActive),
      enabled: Boolean(this.canPeek() && this.hasPossibleMove()),
      meta: this.remainingPeekCount(),
    }];
    return hud;
  }

  getInfoText() {
    return this.variant === 1
      ? 'Show only empty spaces and newly generated tiles.'
      : 'Not sure what newly generated tiles are unless a merge occurs.';
  }
}

export class IceAgeEngine extends BaseMinigameEngine {
  constructor(definition, difficulty, snapshot = null, runtime = null) {
    super(definition, difficulty, snapshot, { deferSetup: true, runtime });
    this.frozenStep = 80 + this.difficulty * 20;
    this.countDown = createBoard(4, 4, 0);
    this.movementTrack = createBoard(4, 4, false);
    this.initialize(snapshot);
  }

  loadLegacyExtra(extra) {
    this.countDown = Array.isArray(extra?.[0]?.[0])
      ? extra[0].map((row) => row.map((value) => Number(value) || 0))
      : createBoard(this.rows, this.cols, 0);
  }

  exportLegacyExtra() {
    return [this.countDown.map((row) => row.slice())];
  }

  setupNewGame() {
    this.countDown = createBoard(this.rows, this.cols, 0);
    super.setupNewGame();
  }

  trackMovementLine(line, reverse = false) {
    const current = reverse ? line.slice().reverse() : line.slice();
    const result = new Array(current.length).fill(false);
    const segments = [];
    let segment = [];
    for (const value of current) {
      if (value === -1) {
        if (segment.length) segments.push(segment);
        segments.push([-1]);
        segment = [];
      } else {
        segment.push(value);
      }
    }
    if (segment.length) segments.push(segment);
    let resultIndex = 0;
    for (const values of segments) {
      if (values.length === 1 && values[0] === -1) {
        result[resultIndex] = false;
        resultIndex += 1;
        continue;
      }
      let moved = false;
      for (let index = 0; index < values.length; index += 1) {
        if (moved) {
          result[resultIndex + index] = false;
          continue;
        }
        if (values[index] === 0) moved = true;
        let step = 1;
        while (index + step < values.length) {
          if (values[index] === values[index + step]) {
            moved = true;
            break;
          }
          if (values[index + step] !== 0) break;
          step += 1;
        }
        result[resultIndex + index] = !moved;
      }
      resultIndex += values.length;
    }
    return reverse ? result.reverse() : result;
  }

  beforeMove(direct) {
    this.movementTrack = createBoard(this.rows, this.cols, false);
    if (direct === 1 || direct === 2) {
      for (let row = 0; row < this.rows; row += 1) {
        this.movementTrack[row] = this.trackMovementLine(this.board[row], direct === 2);
      }
    } else {
      for (let col = 0; col < this.cols; col += 1) {
        const line = this.board.map((row) => row[col]);
        const tracked = this.trackMovementLine(line, direct === 4);
        for (let row = 0; row < this.rows; row += 1) this.movementTrack[row][col] = tracked[row];
      }
    }
  }

  buildStageReveal(row, col, sprite) {
    return { type: 'ice_stage_reveal', index: row * this.cols + col, sprite, durationMs: 280, animDurationMs: 220 };
  }

  spriteForThreshold(previous, current) {
    const thresholds = [
      [20, 'crystal1.png'],
      [36 + this.difficulty * 4, 'crystal3.png'],
      [50 + this.difficulty * 10, 'crystal2.png'],
      [64 + this.difficulty * 16, 'ice_overlay.png'],
      [this.frozenStep - 5, 'icetrap0.png'],
      [this.frozenStep, 'icetrap.png'],
    ];
    const entry = thresholds.find(([threshold]) => previous < threshold && threshold <= current);
    return entry?.[1] || null;
  }

  beforeGenNum() {
    const effects = [];
    for (let row = 0; row < this.rows; row += 1) {
      for (let col = 0; col < this.cols; col += 1) {
        const value = this.board[row][col];
        if (value === -1) continue;
        if (!this.movementTrack[row][col]) {
          this.countDown[row][col] = 0;
          continue;
        }
        const previous = this.countDown[row][col];
        this.countDown[row][col] += 1;
        if (this.countDown[row][col] >= this.frozenStep) {
          this.countDown[row][col] = value;
          this.board[row][col] = -1;
          effects.push(this.buildStageReveal(row, col, 'icetrap.png'));
          continue;
        }
        const sprite = this.spriteForThreshold(previous, this.countDown[row][col]);
        if (sprite) effects.push(this.buildStageReveal(row, col, sprite));
      }
    }
    if (effects.length) this.queueMoveEffects(effects);
  }

  buildViewState() {
    const view = super.buildViewState();
    const tileTextOverride = {};
    const tileStyleVariant = {};
    const coverSprites = {};
    for (let row = 0; row < this.rows; row += 1) {
      for (let col = 0; col < this.cols; col += 1) {
        const index = row * this.cols + col;
        const value = this.board[row][col];
        const countDown = this.countDown[row]?.[col] || 0;
        if (countDown === 0 && value !== -1) continue;
        const sprites = [];
        if (countDown >= 20 || value === -1) sprites.push('crystal1.png');
        if (countDown >= 36 + this.difficulty * 4 || value === -1) sprites.push('crystal3.png');
        if (countDown >= 50 + this.difficulty * 10 || value === -1) sprites.push('crystal2.png');
        if (countDown >= 64 + this.difficulty * 16 || value === -1) sprites.push('ice_overlay.png');
        if (countDown >= this.frozenStep - 5 || value === -1) sprites.push('icetrap0.png');
        if (countDown >= this.frozenStep || value === -1) sprites.push('icetrap.png');
        if (sprites.length) coverSprites[String(index)] = sprites;
        if (value === -1 && countDown > 0) {
          tileTextOverride[String(index)] = String(2 ** countDown);
          tileStyleVariant[String(index)] = { kind: 'frozen', exponent: countDown };
        }
      }
    }
    view.tileTextOverride = tileTextOverride;
    view.tileStyleVariant = tileStyleVariant;
    view.coverSprites = coverSprites;
    return view;
  }

  getInfoText() {
    return 'Tiles freeze in place if they stand still for too long.';
  }
}

export class IsolatedIslandEngine extends BaseMinigameEngine {
  async genNewNum() {
    const spawnChance = 0.04 - countCells(this.board, (value) => value === -3) * 0.02 + this.difficulty * 0.01;
    const positions = emptyPositions(this.board);
    if (positions.length && this.runtime.random() < spawnChance) {
      const [row, col] = randomChoice(positions, this.runtime.rng);
      this.board[row][col] = -3;
      this.newtilePos = row * this.cols + col;
      this.newtile = -3;
      return;
    }
    await super.genNewNum();
  }

  buildViewState() {
    const view = super.buildViewState();
    const coverSprites = {};
    const tileStyleVariant = {};
    flattenBoard(this.board).forEach((value, index) => {
      if (value === -3) {
        coverSprites[String(index)] = ['portal.png'];
        tileStyleVariant[String(index)] = { kind: 'portal' };
      }
    });
    view.coverSprites = coverSprites;
    view.tileStyleVariant = tileStyleVariant;
    return view;
  }

  getInfoText() {
    return 'Discovered a tile that merges with none but itself.';
  }
}

export class ShapeShifterEngine extends BaseMinigameEngine {
  constructor(definition, difficulty, snapshot = null, runtime = null) {
    super(definition, difficulty, snapshot, { deferSetup: true, runtime });
    this.n = 12;
    this.initialize(snapshot);
  }

  getInitialShape(snapshot = null) {
    const board = snapshot?.board;
    if (Array.isArray(board?.[0])) {
      return [board.length, board[0].length];
    }
    return [12, 12];
  }

  selectConnectedCells(n, m) {
    const board = createBoard(n, n, -1);
    const key = (row, col) => `${row},${col}`;
    const visited = new Set();
    const remaining = new Set();
    const neighbors = (row, col) => randomSample(
      [[row - 1, col], [row + 1, col], [row, col - 1], [row, col + 1]],
      4,
      this.runtime.rng
    );
    let stack = [[this.runtime.randomIndex(n), this.runtime.randomIndex(n)]];
    while (visited.size < m) {
      if (!stack.length) {
        const next = randomChoice(Array.from(remaining), this.runtime.rng);
        remaining.delete(next);
        stack = [next.split(',').map(Number)];
      }
      const [row, col] = stack.pop();
      if (row < 0 || row >= n || col < 0 || col >= n || visited.has(key(row, col))) continue;
      visited.add(key(row, col));
      board[row][col] = 0;
      for (const [nextRow, nextCol] of neighbors(row, col)) {
        if (nextRow < 0 || nextRow >= n || nextCol < 0 || nextCol >= n || visited.has(key(nextRow, nextCol))) continue;
        if (this.runtime.random() > 0.5) stack.push([nextRow, nextCol]);
        else remaining.add(key(nextRow, nextCol));
      }
    }
    return board;
  }

  largestRectangleArea(heights) {
    const stack = [];
    let maxArea = 0;
    const extended = heights.concat(0);
    for (let index = 0; index < extended.length; index += 1) {
      while (stack.length && extended[index] < extended[stack[stack.length - 1]]) {
        const height = extended[stack.pop()];
        const width = stack.length ? index - stack[stack.length - 1] - 1 : index;
        maxArea = Math.max(maxArea, height * width);
      }
      stack.push(index);
    }
    return maxArea;
  }

  maxRectangleArea(matrix) {
    if (!flattenBoard(matrix).some((value) => value === 0)) return 0;
    const rows = matrix.length;
    const cols = matrix[0].length;
    const heights = new Array(cols).fill(0);
    let maxArea = 0;
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        heights[col] = matrix[row][col] === 0 ? heights[col] + 1 : 0;
      }
      maxArea = Math.max(maxArea, this.largestRectangleArea(heights));
    }
    return maxArea;
  }

  minBoundingRectangle(matrix) {
    let minRow = matrix.length;
    let maxRow = 0;
    let minCol = matrix[0].length;
    let maxCol = 0;
    for (let row = 0; row < matrix.length; row += 1) {
      for (let col = 0; col < matrix[row].length; col += 1) {
        if (matrix[row][col] === 0) {
          minRow = Math.min(minRow, row);
          maxRow = Math.max(maxRow, row);
          minCol = Math.min(minCol, col);
          maxCol = Math.max(maxCol, col);
        }
      }
    }
    return matrix.slice(minRow, maxRow + 1).map((row) => row.slice(minCol, maxCol + 1));
  }

  transpose(matrix) {
    return Array.from({ length: matrix[0].length }, (_value, row) => matrix.map((sourceRow) => sourceRow[row]));
  }

  setupNewGame() {
    let rectangleArea = 10000;
    let board = createBoard(this.n, this.n, -1);
    while (rectangleArea > 13 - this.difficulty * 2 || rectangleArea < 8 - this.difficulty * 2) {
      board = this.selectConnectedCells(this.n, 18);
      rectangleArea = this.maxRectangleArea(board);
    }
    board = this.minBoundingRectangle(board);
    if (board[0].length < board.length) board = this.transpose(board);
    this.rows = board.length;
    this.cols = board[0].length;
    let spawn = genNewNum(board, SPAWN_RATE4, this.runtime.rng);
    spawn = genNewNum(spawn.board, SPAWN_RATE4, this.runtime.rng);
    this.board = spawn.board;
    this.newtilePos = spawn.index;
    this.newtile = spawn.value;
    this.score = 0;
    this.isOver = false;
    this.currentMaxNum = positiveMax(this.board);
    this.pendingMessages = {};
    this.animation = new this.animation.constructor({ appearIndex: this.newtilePos, appearValue: this.newtile, validMove: true });
  }

  buildViewState() {
    const view = super.buildViewState();
    view.blockedMask = flattenBoard(this.board).map((value) => value === -1);
    return view;
  }

  getInfoText() {
    return 'Every game begins with a unique and unexpected board shape!';
  }
}

export class BlitzkriegEngine extends BaseMinigameEngine {
  constructor(definition, difficulty, snapshot = null, runtime = null) {
    super(definition, difficulty, snapshot, { deferSetup: true, runtime });
    this.remainingMs = 180000;
    this.timerRunning = false;
    this.timerAnchorMs = null;
    this.count1k = 0;
    this.pendingBonusMs = 0;
    this.initialize(snapshot);
    this.count1k = countCells(this.board, (value) => value === 10);
  }

  loadLegacyExtra(extra) {
    if (extra?.length) this.remainingMs = Math.max(0, Number(extra[0]) * 60 * 1000);
    this.timerRunning = Boolean(extra?.[1]) && this.remainingMs > 0;
    this.timerAnchorMs = this.timerRunning ? this.runtime.now() : null;
  }

  exportLegacyExtra() {
    this.syncTimer();
    return [this.remainingMs / (60 * 1000), this.timerRunning];
  }

  setupNewGame() {
    super.setupNewGame();
    this.remainingMs = 180000;
    this.timerRunning = false;
    this.timerAnchorMs = null;
    this.count1k = countCells(this.board, (value) => value === 10);
    this.pendingBonusMs = 0;
    this.isOver = false;
  }

  syncTimer() {
    if (!this.timerRunning || this.timerAnchorMs == null || this.isOver) return;
    const now = this.runtime.now();
    const elapsed = Math.max(0, now - this.timerAnchorMs);
    this.remainingMs = Math.max(0, this.remainingMs - elapsed);
    this.timerAnchorMs = now;
    if (this.remainingMs === 0) {
      this.isOver = true;
      this.timerRunning = false;
      this.checkGamePassed();
    }
  }

  async doMove(direction) {
    this.syncTimer();
    if (this.isOver) return;
    if (!this.timerRunning) {
      this.timerRunning = true;
      this.timerAnchorMs = this.runtime.now();
    }
    const previousCount = countCells(this.board, (value) => value === 10);
    await super.doMove(direction);
    this.syncTimer();
    if (!this.lastValidMove) return;
    const currentCount = countCells(this.board, (value) => value === 10);
    if (currentCount > previousCount) {
      const bonusMinutes = this.difficulty === 1 ? 0.75 : 1.0;
      const gained = Math.trunc((currentCount - previousCount) * bonusMinutes * 60 * 1000);
      this.remainingMs += gained;
      this.pendingBonusMs = gained;
      if (this.timerRunning) this.timerAnchorMs = this.runtime.now();
    }
    this.count1k = currentCount;
    if (this.isOver) this.timerRunning = false;
  }

  checkGamePassed() {
    this.refreshHighestTileExp();
    this.currentMaxNum = Math.max(this.currentMaxNum, positiveMax(this.board));
    this.maxNum = Math.max(this.maxNum, this.currentMaxNum);
    if (!this.isOver) return;
    if (this.maxNum > 9) this.isPassed = { 12: 3, 11: 2, 10: 1 }[this.maxNum] || 4;
    if (this.currentMaxNum <= 9) return;
    const level = { 12: 'gold', 11: 'silver', 10: 'bronze' }[this.currentMaxNum] || 'gold';
    const message = this.score === this.maxScore
      ? `You achieved ${this.score} score! You get a ${level} trophy!`
      : `You achieved ${this.score} score! Nice game!`;
    this.queueMessage('trophy', { level, message });
  }

  checkGameOver() {
    this.syncTimer();
    if (!this.isOver && this.remainingMs > 0 && this.hasPossibleMove()) return;
    this.isOver = true;
    this.timerRunning = false;
    this.checkGamePassed();
  }

  buildHud() {
    this.syncTimer();
    const hud = super.buildHud();
    const countdown = {
      type: 'countdown',
      title: 'Countdown',
      remainingMs: Math.trunc(this.remainingMs),
      running: Boolean(this.timerRunning && !this.isOver),
      syncedAt: this.runtime.now(),
    };
    if (this.pendingBonusMs > 0) countdown.bonusMs = this.pendingBonusMs;
    hud.customPanels = [countdown];
    return hud;
  }

  clearAnimation() {
    super.clearAnimation();
    this.pendingBonusMs = 0;
  }

  getInfoText() {
    return 'Act quickly to earn bonus time and rack up the highest score!';
  }
}

export class TrickyTilesEngine extends BaseMinigameEngine {
  constructor(definition, difficulty, snapshot = null, runtime = null) {
    super(definition, difficulty, snapshot, { deferSetup: true, runtime });
    this.evilGenProb = 0.33 + this.difficulty * 0.1;
    this.initialize(snapshot);
  }

  loadLegacyExtra(extra) {
    const savedProbability = Number(extra?.[0]);
    if (Number.isFinite(savedProbability) && savedProbability > 0 && savedProbability < 1) {
      this.evilGenProb = savedProbability;
    }
  }

  exportLegacyExtra() {
    return [this.evilGenProb];
  }

  setupNewGame() {
    super.setupNewGame();
    this.evilGenProb = 0.33 + this.runtime.random() / 12 + this.difficulty * 0.1;
  }

  async genNewNum() {
    if (this.runtime.random() > this.evilGenProb) {
      await super.genNewNum();
      return;
    }
    const emptyCount = countCells(this.board, (value) => value === 0);
    const spawn = await (this.runtime.evilSpawn || generateEvilSpawn)(this.board, emptyCount < 6 ? 5 : 4);
    if (!spawn || this.board[Math.floor(spawn.index / this.cols)]?.[spawn.index % this.cols] !== 0) {
      this.runtime.markDeterminismFailure?.('evil_spawn_unavailable');
      await super.genNewNum();
      return;
    }
    this.board[Math.floor(spawn.index / this.cols)][spawn.index % this.cols] = spawn.value;
    this.newtilePos = spawn.index;
    this.newtile = spawn.value;
  }

  getInfoText() {
    return 'Brace yourself! New numbers may appear in the most challenging spots!';
  }
}

export class EndlessFamilyEngine extends BaseMinigameEngine {
  constructor(definition, difficulty, snapshot = null, runtime = null) {
    super(definition, difficulty, snapshot, { deferSetup: true, runtime });
    this.variant = this.detectVariant(definition.legacyName);
    this.bombPos = null;
    this.currentLevel = 0;
    this.bombType = 0;
    this.targetPos = null;
    this.countDown = createBoard(4, 4, 0);
    this.hasJustExploded = false;
    this.pendingResolutionKind = null;
    this.pendingResolutionPositions = [];
    this.levels = this.variant === 'hybrid'
      ? [[150000, 4, null], [100000, 3, 'gold'], [50000, 2, 'silver'], [20000, 1, 'bronze']]
      : [[300000, 4, null], [200000, 3, 'gold'], [100000, 2, 'silver'], [40000, 1, 'bronze']];
    this.bombGenRate = this.variant === 'hybrid' ? 0.05 : 0.03;
    this.initialize(snapshot);
  }

  detectVariant(name) {
    return {
      'Endless Explosions': 'explosions',
      'Endless Giftbox': 'giftbox',
      'Endless Factorization': 'factorization',
      'Endless Hybrid': 'hybrid',
      'Endless AirRaid': 'airraid',
    }[name] || 'explosions';
  }

  loadLegacyExtra(extra) {
    if (this.variant === 'airraid') {
      this.countDown = Array.isArray(extra?.[0]?.[0]) ? extra[0].map((row) => row.map(Number)) : createBoard(this.rows, this.cols, 0);
      this.targetPos = Array.isArray(extra?.[1]) ? extra[1].map(Number) : null;
      this.currentLevel = Number(extra?.[2] || 0);
      return;
    }
    this.bombPos = Array.isArray(extra?.[0]) ? extra[0].map(Number) : null;
    this.currentLevel = Number(extra?.[1] || 0);
    if (this.variant === 'hybrid') this.bombType = Number(extra?.[2] || 0);
  }

  exportLegacyExtra() {
    if (this.variant === 'airraid') {
      return [this.countDown.map((row) => row.slice()), this.targetPos ? this.targetPos.slice() : null, this.currentLevel];
    }
    const data = [this.bombPos ? this.bombPos.slice() : null, this.currentLevel];
    if (this.variant === 'hybrid') data.push(this.bombType);
    return data;
  }

  setupNewGame() {
    this.hasJustExploded = false;
    this.pendingResolutionKind = null;
    this.pendingResolutionPositions = [];
    this.currentLevel = 0;
    this.bombPos = null;
    this.targetPos = null;
    this.countDown = createBoard(this.rows, this.cols, 0);
    if (this.variant === 'hybrid') this.bombType = this.runtime.randomIndex(3);
    super.setupNewGame();
    if (this.variant === 'airraid') {
      this.targetPos = null;
      this.countDown = createBoard(this.rows, this.cols, 0);
    } else {
      this.bombPos = randomChoice(emptyPositions(this.board), this.runtime.rng);
    }
  }

  queueScoreTrophy() {
    let levelName = null;
    for (const [threshold, level, trophy] of this.levels) {
      if (this.score >= threshold && this.currentLevel < level) {
        this.isPassed = Math.max(this.isPassed, level);
        this.currentLevel = level;
        levelName = trophy;
        break;
      }
    }
    if (!levelName) return;
    const scoreText = `${Math.floor(this.maxScore / 1000)}k`;
    this.queueMessage('trophy', {
      level: levelName,
      message: this.maxScore === this.score
        ? `You achieved ${scoreText} score! You get a ${levelName} trophy!`
        : `You achieved ${scoreText} score! Let's go!`,
    });
  }

  checkGamePassed() {
    this.refreshHighestTileExp();
    this.currentMaxNum = Math.max(this.currentMaxNum, positiveMax(this.board));
    this.maxNum = Math.max(this.maxNum, this.currentMaxNum);
    this.queueScoreTrophy();
  }

  hasPossibleMove() {
    if (this.variant !== 'airraid' && this.bombPos) return true;
    return super.hasPossibleMove();
  }

  placeRandomBomb() {
    const positions = emptyPositions(this.board);
    if (!positions.length) return false;
    this.bombPos = randomChoice(positions, this.runtime.rng);
    if (this.variant === 'hybrid') this.bombType = this.runtime.randomIndex(3);
    return true;
  }

  bombSlideDistance(direct) {
    if (!this.bombPos) return null;
    const [row, col] = this.bombPos;
    if (direct === 1) return [row, 0];
    if (direct === 2) return [row, this.cols - 1];
    if (direct === 3) return [0, col];
    if (direct === 4) return [this.rows - 1, col];
    return null;
  }

  firstHitInDirection(direct) {
    if (!this.bombPos) return null;
    const [row, col] = this.bombPos;
    let search = [];
    if (direct === 1) search = Array.from({ length: col }, (_v, index) => [row, col - 1 - index]).concat(Array.from({ length: this.cols - col - 1 }, (_v, index) => [row, col + 1 + index]));
    if (direct === 2) search = Array.from({ length: this.cols - col - 1 }, (_v, index) => [row, col + 1 + index]).concat(Array.from({ length: col }, (_v, index) => [row, col - 1 - index]));
    if (direct === 3) search = Array.from({ length: row }, (_v, index) => [row - 1 - index, col]).concat(Array.from({ length: this.rows - row - 1 }, (_v, index) => [row + 1 + index, col]));
    if (direct === 4) search = Array.from({ length: this.rows - row - 1 }, (_v, index) => [row + 1 + index, col]).concat(Array.from({ length: row }, (_v, index) => [row - 1 - index, col]));
    return search.find(([candidateRow, candidateCol]) => this.board[candidateRow][candidateCol] !== 0) || null;
  }

  currentObjectVisual() {
    if (this.variant === 'explosions') return ['bomb.png', ''];
    if (this.variant === 'giftbox') return ['giftbox.png', ''];
    if (this.variant === 'factorization') return ['tilebg.png', ''];
    if (this.variant === 'hybrid') return this.bombType === 0 ? ['bomb.png', ''] : this.bombType === 1 ? ['tilebg.png', ''] : ['giftbox.png', ''];
    return ['bomb.png', ''];
  }

  queueObjectSlide(fromPos, toPos, { hideTarget = false, fadeOutAtEnd = false, durationMs = 100 } = {}) {
    if (!fromPos || !toPos || (fromPos[0] === toPos[0] && fromPos[1] === toPos[1] && !hideTarget)) return;
    const [sprite, labelText] = this.currentObjectVisual();
    const effect = {
      type: 'object_slide',
      fromIndex: fromPos[0] * this.cols + fromPos[1],
      toIndex: toPos[0] * this.cols + toPos[1],
      durationMs,
      animDurationMs: durationMs,
      sprite,
      labelText,
      fadeOutAtEnd,
    };
    if (hideTarget) effect.hideIndices = [toPos[0] * this.cols + toPos[1]];
    this.queueMoveEffects([effect]);
  }

  explodeTarget(row, col) {
    const value = this.board[row][col];
    const markSingle = (kind) => {
      this.hasJustExploded = value;
      this.board[row][col] = -2;
      this.pendingResolutionKind = kind;
      this.pendingResolutionPositions = [[row, col]];
    };
    if (this.variant === 'explosions' || (this.variant === 'hybrid' && this.bombType === 0)) {
      this.board[row][col] = 0;
      this.queueMoveEffects([{ type: 'explosion', index: row * this.cols + col, delayMs: 100, durationMs: 500, animDurationMs: 500 }]);
      this.hasJustExploded = false;
      return;
    }
    if (this.variant === 'giftbox' || (this.variant === 'hybrid' && this.bombType === 2)) {
      markSingle('giftbox_burst');
      return;
    }
    markSingle('factorization_burst');
    if (value > 1 && this.bombPos) {
      this.board[this.bombPos[0]][this.bombPos[1]] = -2;
      this.pendingResolutionPositions.push(this.bombPos.slice());
    } else if (this.bombPos) {
      this.board[this.bombPos[0]][this.bombPos[1]] = 0;
    }
  }

  resolvePostExplosion() {
    if (!this.hasJustExploded) return;
    const positions = [];
    for (let row = 0; row < this.rows; row += 1) {
      for (let col = 0; col < this.cols; col += 1) {
        if (this.board[row][col] === -2) positions.push([row, col]);
      }
    }
    const original = Number(this.hasJustExploded);
    if (this.variant === 'giftbox' || (this.variant === 'hybrid' && this.bombType === 2)) {
      if (positions.length) {
        const [row, col] = positions[0];
        let value = original;
        while (value === original) value = weightedExponentChoice(this.runtime.rng);
        this.board[row][col] = value;
        this.queueMoveEffects([{ type: 'giftbox_burst', index: row * this.cols + col, delayMs: 100, durationMs: 620, animDurationMs: 500, hideIndices: [row * this.cols + col] }]);
      }
      this.hasJustExploded = false;
      this.pendingResolutionKind = null;
      this.pendingResolutionPositions = [];
      return;
    }
    if (original <= 1) {
      if (positions.length) {
        const [row, col] = positions[0];
        this.board[row][col] = 0;
        this.queueMoveEffects([{ type: 'factorization_burst', index: row * this.cols + col, delayMs: 100, durationMs: 620, animDurationMs: 500, hideIndices: [row * this.cols + col] }]);
      }
    } else if (positions.length >= 2) {
      const [[row0, col0], [row1, col1]] = positions;
      const factor1 = 1 + this.runtime.randomIndex(Math.max(original - 1, 1));
      const factor0 = original - factor1;
      this.board[row0][col0] = factor0;
      this.board[row1][col1] = factor1;
      this.queueMoveEffects([
        { type: 'factorization_burst', index: row0 * this.cols + col0, delayMs: 100, durationMs: 620, animDurationMs: 500, hideIndices: [row0 * this.cols + col0] },
        { type: 'factorization_burst', index: row1 * this.cols + col1, delayMs: 100, durationMs: 620, animDurationMs: 500, hideIndices: [row1 * this.cols + col1] },
      ]);
    }
    this.hasJustExploded = false;
    this.pendingResolutionKind = null;
    this.pendingResolutionPositions = [];
  }

  beforeGenNum() {
    if (this.variant === 'airraid') {
      for (let row = 0; row < this.rows; row += 1) {
        for (let col = 0; col < this.cols; col += 1) {
          if (this.countDown[row][col] > 0) {
            this.countDown[row][col] -= 1;
            if (this.countDown[row][col] === 0) this.board[row][col] = 0;
          }
        }
      }
      if (this.targetPos && this.board[this.targetPos[0]][this.targetPos[1]] !== 0) {
        const [row, col] = this.targetPos;
        this.board[row][col] = -1;
        this.countDown[row][col] = 60 + this.difficulty * 40;
        this.queueMoveEffects([
          { type: 'airraid_fire_drop', index: row * this.cols + col, durationMs: 250, animDurationMs: 250 },
          { type: 'airraid_explosion', index: row * this.cols + col, delayMs: 250, durationMs: 450, animDurationMs: 200 },
        ]);
      }
      this.targetPos = null;
      return;
    }
    this.resolvePostExplosion();
  }

  async genNewNum() {
    if (this.variant === 'airraid') {
      const spawn = genNewNum(this.board, SPAWN_RATE4, this.runtime.rng);
      this.board = spawn.board;
      this.newtilePos = spawn.index;
      this.newtile = spawn.value;
      const probability = Math.max(0.08 - countCells(this.countDown, (value) => value > 0) / 40, 0.01);
      if (spawn.emptyCount > 1 && !this.targetPos && this.runtime.random() < probability) {
        const positions = emptyPositions(this.board);
        if (positions.length) this.targetPos = randomChoice(positions, this.runtime.rng);
      }
      return;
    }
    if (!this.bombPos && this.runtime.random() < this.bombGenRate) {
      if (this.placeRandomBomb()) {
        this.newtilePos = this.bombPos[0] * this.cols + this.bombPos[1];
        this.newtile = 0;
        return;
      }
    }
    let boardForSpawn = cloneBoard(this.board);
    if (this.bombPos) boardForSpawn[this.bombPos[0]][this.bombPos[1]] = 1;
    const spawn = genNewNum(boardForSpawn, SPAWN_RATE4, this.runtime.rng);
    if (this.bombPos) spawn.board[this.bombPos[0]][this.bombPos[1]] = 0;
    if (this.variant === 'hybrid') {
      const chance = 0.03 - countCells(spawn.board, (value) => value === -3) * 0.02 + this.difficulty * 0.015;
      const positions = emptyPositions(spawn.board);
      if (positions.length && this.runtime.random() < chance) {
        const [row, col] = randomChoice(positions, this.runtime.rng);
        spawn.board[row][col] = -3;
        this.newtilePos = row * this.cols + col;
        this.newtile = -3;
        this.board = spawn.board;
        return;
      }
    }
    this.board = spawn.board;
    this.newtilePos = spawn.index;
    this.newtile = spawn.value;
  }

  objectSlideTarget(direct) {
    if (!this.bombPos) return [-1, -1];
    const [row, col] = this.bombPos;
    if (direct === 1) return [row, col - countZeros(this.board[row].slice(0, col).reverse())];
    if (direct === 2) return [row, col + countZeros(this.board[row].slice(col + 1))];
    if (direct === 3) return [row - countZeros(this.board.slice(0, row).map((sourceRow) => sourceRow[col]).reverse()), col];
    if (direct === 4) return [row + countZeros(this.board.slice(row + 1).map((sourceRow) => sourceRow[col])), col];
    return [row, col];
  }

  async moveAndCheckValidity(direct) {
    if (this.variant === 'airraid') {
      return super.moveAndCheckValidity(direct);
    }
    let valid = false;
    if (this.bombPos) {
      const original = this.bombPos.slice();
      const slideTarget = this.objectSlideTarget(direct);
      const hit = this.firstHitInDirection(direct);
      if (hit) {
        this.queueObjectSlide(original, slideTarget, { fadeOutAtEnd: true });
        this.explodeTarget(hit[0], hit[1]);
        this.bombPos = null;
        valid = true;
      } else {
        const target = this.bombSlideDistance(direct);
        if (target && (target[0] !== this.bombPos[0] || target[1] !== this.bombPos[1])) {
          this.queueObjectSlide(original, target, { hideTarget: true });
          this.bombPos = target;
          valid = true;
        }
      }
    }
    const result = moveBoard(this.board, ['left', 'right', 'up', 'down'][direct - 1]);
    return { board: result.board, score: result.score, valid: valid || result.valid };
  }

  buildViewState() {
    const view = super.buildViewState();
    const coverSprites = {};
    const tileTextOverride = {};
    const tileStyleVariant = {};
    const blockedMask = view.blockedMask.slice();

    if (this.variant === 'airraid') {
      for (let row = 0; row < this.rows; row += 1) {
        for (let col = 0; col < this.cols; col += 1) {
          const index = row * this.cols + col;
          if (this.targetPos && this.targetPos[0] === row && this.targetPos[1] === col) {
            coverSprites[String(index)] = ['target.png'];
          }
          if (this.board[row][col] === -1) {
            blockedMask[index] = true;
            coverSprites[String(index)] = [this.countDown[row][col] > (60 + this.difficulty * 40) / 2 ? 'crater1.png' : 'crater2.png'];
          }
        }
      }
      view.blockedMask = blockedMask;
      view.coverSprites = coverSprites;
      return view;
    }

    if (this.bombPos) {
      const index = this.bombPos[0] * this.cols + this.bombPos[1];
      coverSprites[String(index)] = [this.currentObjectVisual()[0]];
    }
    if (this.variant === 'hybrid') {
      flattenBoard(this.board).forEach((value, index) => {
        if (value === -3) {
          coverSprites[String(index)] = ['portal.png'];
          tileStyleVariant[String(index)] = { kind: 'portal' };
        }
      });
    }
    view.coverSprites = coverSprites;
    view.tileTextOverride = tileTextOverride;
    view.tileStyleVariant = tileStyleVariant;
    return view;
  }

  getInfoText() {
    return {
      explosions: 'A small chance of generating a bomb that destroys the first tile it encounters!',
      giftbox: 'A small chance of generating a gift box that magically changes the first tile it encounters!',
      factorization: 'A small chance of generating a power-up that halves the first tile it encounters!',
      hybrid: 'All-Stars.',
      airraid: 'Airstrikes incoming! Avoid marked targets!',
    }[this.variant];
  }
}

export const ENGINE_BY_MODULE = {
  design_master: DesignMasterEngine,
  gravity_twist: GravityTwistEngine,
  column_chaos: ColumnChaosEngine,
  blitzkrieg: BlitzkriegEngine,
  mystery_merge: MysteryMergeEngine,
  ice_age: IceAgeEngine,
  isolated_island: IsolatedIslandEngine,
  shape_shifter: ShapeShifterEngine,
  endless_family: EndlessFamilyEngine,
  ferris_wheel: FerrisWheelEngine,
  tricky_tiles: TrickyTilesEngine,
};

export function createEngine(definition, difficulty, snapshot = null, runtime = null) {
  const Engine = ENGINE_BY_MODULE[definition.moduleKey];
  if (!Engine) {
    throw new Error(`Minigame '${definition.title}' is not implemented yet.`);
  }
  return new Engine(definition, difficulty, snapshot, runtime);
}
