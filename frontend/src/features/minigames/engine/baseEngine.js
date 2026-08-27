import { buildMoveAnimationMetadata, genNewNum, moveBoard } from './boardMover.js';
import {
  boardFromFlat,
  boardShape,
  cloneBoard,
  codeToDirection,
  createBoard,
  directionToCode,
  flattenBoard,
  positiveMax,
  sanitizeExtra,
  SPAWN_RATE4,
} from './utils.js';
import { createMinigameRuntime } from './runtime.js';

export const TROPHY_LEVEL_NAMES = Object.freeze({
  1: 'bronze',
  2: 'silver',
  3: 'gold',
  4: 'grand',
});

export function trophyLevelForExponent(exponent, goldExponent) {
  const normalizedExponent = Math.trunc(Number(exponent) || 0);
  const normalizedGoldExponent = Math.trunc(Number(goldExponent) || 0);
  if (normalizedExponent > normalizedGoldExponent) return 4;
  return {
    [normalizedGoldExponent]: 3,
    [normalizedGoldExponent - 1]: 2,
    [normalizedGoldExponent - 2]: 1,
  }[normalizedExponent] || 0;
}

export function trophyLevelName(level) {
  return TROPHY_LEVEL_NAMES[Math.trunc(Number(level) || 0)] || '';
}

export class AnimationState {
  constructor({
    appearIndex = null,
    appearValue = null,
    direction = null,
    validMove = false,
    slideDistances = null,
    popPositions = null,
    effects = null,
    pageEffects = null,
    followUp = null,
  } = {}) {
    this.appearIndex = appearIndex;
    this.appearValue = appearValue;
    this.direction = direction;
    this.validMove = validMove;
    this.slideDistances = slideDistances;
    this.popPositions = popPositions;
    this.effects = effects;
    this.pageEffects = pageEffects;
    this.followUp = followUp;
  }

  toPayload() {
    if (!this.validMove && !this.effects?.length && !this.pageEffects?.length) {
      return this.followUp ? { followUp: { ...this.followUp } } : {};
    }
    const payload = {
      direction: this.direction,
      slide_distances: Array.isArray(this.slideDistances) ? this.slideDistances.slice() : [],
      pop_positions: Array.isArray(this.popPositions) ? this.popPositions.slice() : [],
    };
    if (this.appearIndex != null && this.appearValue != null) {
      payload.appearTile = {
        index: Number(this.appearIndex),
        value: Number(this.appearValue),
      };
    }
    if (this.effects?.length) {
      payload.effects = this.effects.map((effect) => ({ ...effect }));
    }
    if (this.pageEffects?.length) {
      payload.pageEffects = this.pageEffects.map((effect) => ({ ...effect }));
    }
    if (this.followUp) {
      payload.followUp = { ...this.followUp };
    }
    return payload;
  }
}

export class BaseMinigameEngine {
  constructor(definition, difficulty, snapshot = null, options = {}) {
    const directRuntime = options?.rng && options?.clock ? options : null;
    const deferSetup = directRuntime ? false : Boolean(options?.deferSetup);
    const runtime = directRuntime || options?.runtime || null;
    this.definition = definition;
    this.gameId = definition.id;
    this.title = definition.title;
    this.legacyName = definition.legacyName;
    this.difficulty = Number(difficulty) ? 1 : 0;
    this.runtime = runtime || createMinigameRuntime();

    const [rows, cols] = this.getInitialShape(snapshot);
    this.rows = rows;
    this.cols = cols;
    this.board = createBoard(rows, cols);

    this.score = 0;
    this.maxScore = 0;
    this.maxNum = 0;
    this.currentMaxNum = 0;
    this.highestTileExp = 0;
    this.isPassed = 0;
    this.newtilePos = -1;
    this.newtile = 0;
    this.isOver = false;

    this.pendingMessages = {};
    this.animation = new AnimationState();
    this.queuedFollowUpAnimation = null;
    this.queuedMoveEffects = [];
    this.queuedPageEffects = [];
    this.lastValidMove = false;
    this.lastDirection = null;
    this.lastMoveAtMs = 0;

    if (!deferSetup) {
      this.initialize(snapshot);
    }
  }

  initialize(snapshot = null) {
    if (snapshot && typeof snapshot === 'object') {
      this.importSnapshot(snapshot);
      if (flattenBoard(this.board).every((value) => value <= 0)) {
        this.setupNewGame();
      }
    } else {
      this.setupNewGame();
    }
  }

  getInitialShape(_snapshot = null) {
    return [4, 4];
  }

  setupNewGame() {
    let next = createBoard(this.rows, this.cols);
    let spawn = genNewNum(next, SPAWN_RATE4, this.runtime.rng);
    next = spawn.board;
    spawn = genNewNum(next, SPAWN_RATE4, this.runtime.rng);
    this.board = cloneBoard(spawn.board);
    this.score = 0;
    this.newtilePos = spawn.index;
    this.newtile = spawn.value;
    this.currentMaxNum = positiveMax(this.board);
    this.highestTileExp = this.currentMaxNum;
    this.isOver = false;
    this.pendingMessages = {};
    this.animation = new AnimationState({
      appearIndex: this.newtilePos,
      appearValue: this.newtile,
      validMove: true,
    });
  }

  setupNewRound() {
    this.setupNewGame();
  }

  loadLegacyExtra(_extraState) {}

  exportLegacyExtra() {
    return [];
  }

  refreshHighestTileExp() {
    this.highestTileExp = Math.max(Number(this.highestTileExp) || 0, positiveMax(this.board));
  }

  exportSnapshot() {
    return {
      gameId: this.gameId,
      legacyName: this.legacyName,
      difficulty: this.difficulty,
      board: cloneBoard(this.board),
      score: Number(this.score) || 0,
      maxScore: Number(this.maxScore) || 0,
      maxNum: Number(this.maxNum) || 0,
      currentMaxNum: Number(this.currentMaxNum) || 0,
      isPassed: Number(this.isPassed) || 0,
      newtilePos: Number.isFinite(Number(this.newtilePos)) ? Number(this.newtilePos) : -1,
      newtile: Number(this.newtile) || 0,
      highestTileExp: Number(this.highestTileExp) || 0,
      isOver: Boolean(this.isOver),
      extra: sanitizeExtra(this.exportLegacyExtra()),
    };
  }

  importSnapshot(snapshot) {
    if (!snapshot || typeof snapshot !== 'object') return;
    const rawBoard = snapshot.board;
    const board = Array.isArray(rawBoard?.[0])
      ? cloneBoard(rawBoard)
      : boardFromFlat(rawBoard, this.rows, this.cols);
    const shape = boardShape(board);
    if (shape.rows > 0 && shape.cols > 0) {
      this.rows = shape.rows;
      this.cols = shape.cols;
      this.board = board;
    }
    this.score = Number(snapshot.score ?? this.score) || 0;
    this.maxScore = Number(snapshot.maxScore ?? this.maxScore) || 0;
    this.maxNum = Number(snapshot.maxNum ?? this.maxNum) || 0;
    this.currentMaxNum = Number(snapshot.currentMaxNum ?? this.currentMaxNum) || 0;
    this.isPassed = Number(snapshot.isPassed ?? this.isPassed) || 0;
    this.newtilePos = Number.isFinite(Number(snapshot.newtilePos ?? this.newtilePos))
      ? Number(snapshot.newtilePos ?? this.newtilePos)
      : -1;
    this.newtile = Number(snapshot.newtile ?? this.newtile) || 0;
    this.highestTileExp = Number(snapshot.highestTileExp ?? this.highestTileExp) || 0;
    this.isOver = Boolean(snapshot.isOver ?? this.isOver);
    if (Array.isArray(snapshot.extra)) {
      this.loadLegacyExtra(snapshot.extra);
    }
    this.currentMaxNum = Math.max(this.currentMaxNum, positiveMax(this.board));
    this.refreshHighestTileExp();
  }

  queueMessage(key, value) {
    this.pendingMessages[key] = value;
  }

  popMessages() {
    const payload = { ...this.pendingMessages };
    this.pendingMessages = {};
    return payload;
  }

  async genNewNum() {
    const spawn = genNewNum(this.board, SPAWN_RATE4, this.runtime.rng);
    this.board = cloneBoard(spawn.board);
    this.newtilePos = spawn.index;
    this.newtile = spawn.value;
    this.refreshHighestTileExp();
  }

  beforeMove(_direct) {}

  beforeGenNum(_direct) {}

  afterGenNum() {}

  async moveAndCheckValidity(direct) {
    const direction = codeToDirection(direct);
    const result = moveBoard(this.board, direction);
    return {
      board: result.board,
      score: result.score,
      valid: result.valid,
    };
  }

  hasPossibleMove() {
    if (flattenBoard(this.board).some((value) => value === 0)) {
      return true;
    }
    for (const direction of ['left', 'right', 'up', 'down']) {
      if (moveBoard(this.board, direction).valid) {
        return true;
      }
    }
    return false;
  }

  checkGamePassed() {
    const currentPeak = positiveMax(this.board);
    this.refreshHighestTileExp();
    const previousPeak = Number(this.currentMaxNum) || 0;
    const previousBest = Number(this.maxNum) || 0;
    this.currentMaxNum = Math.max(this.currentMaxNum, currentPeak);
    this.maxNum = Math.max(this.maxNum, this.currentMaxNum);
    if (this.currentMaxNum <= 9) return;
    if (this.currentMaxNum > previousPeak) {
      const trophyLevel = trophyLevelForExponent(this.currentMaxNum, 12);
      const previousTrophyLevel = Number(this.isPassed) || 0;
      this.isPassed = Math.max(previousTrophyLevel, trophyLevel);
      const level = trophyLevelName(trophyLevel);
      let message;
      if (this.currentMaxNum > previousBest) {
        message = trophyLevel > previousTrophyLevel
          ? `You achieved ${2 ** this.maxNum}! You get a ${level} trophy!`
          : `You achieved ${2 ** this.currentMaxNum}! Take it further!`;
      } else {
        message = `You achieved ${2 ** this.currentMaxNum}! Take it further!`;
      }
      this.queueMessage('trophy', { level, message });
    }
  }

  checkGameOver() {
    const wasOver = Boolean(this.isOver);
    this.isOver = !this.hasPossibleMove();
    if (this.isOver && !wasOver) {
      this.queueMessage('gameOver', 'Game Over');
    }
  }

  requestInfo() {
    this.queueMessage('infoDialog', this.getInfoText());
  }

  getInfoText() {
    return 'More minigames.';
  }

  handleCustomAction(_key, _phase = 'trigger') {
    return false;
  }

  clearAnimation() {
    this.animation = new AnimationState();
    this.queuedFollowUpAnimation = null;
    this.queuedMoveEffects = [];
    this.queuedPageEffects = [];
  }

  setSpecialEffects(effects = []) {
    this.animation = new AnimationState({ effects: effects.map((effect) => ({ ...effect })) });
    this.queuedFollowUpAnimation = null;
    this.queuedMoveEffects = [];
    this.queuedPageEffects = [];
  }

  queueMoveEffects(effects = []) {
    if (!Array.isArray(effects) || !effects.length) return;
    this.queuedMoveEffects.push(...effects.map((effect) => ({ ...effect })));
  }

  queuePageEffects(effects = []) {
    if (!Array.isArray(effects) || !effects.length) return;
    this.queuedPageEffects.push(...effects.map((effect) => ({ ...effect })));
  }

  setFollowUpAnimation(animation = null) {
    this.queuedFollowUpAnimation = animation ? { ...animation } : null;
  }

  async doMove(direction) {
    const directionKey = String(direction || '').trim().toLowerCase();
    const direct = directionToCode(directionKey);
    this.lastDirection = direct ? directionKey : null;
    this.lastValidMove = false;
    this.clearAnimation();
    if (!direct) return;

    const boardBefore = cloneBoard(this.board);
    this.beforeMove(direct);
    const moveResult = await this.moveAndCheckValidity(direct);
    if (!moveResult.valid) {
      this.checkGameOver();
      return;
    }

    this.board = cloneBoard(moveResult.board);
    this.score += Number(moveResult.score) || 0;
    this.maxScore = Math.max(this.maxScore, this.score);
    this.beforeGenNum(direct);
    await this.genNewNum();
    this.afterGenNum();
    this.refreshHighestTileExp();
    this.checkGamePassed();
    this.checkGameOver();

    this.lastValidMove = true;
    this.lastMoveAtMs = this.runtime.now();
    const animationMetadata = buildMoveAnimationMetadata(
      boardBefore,
      directionKey,
      this.newtilePos >= 0 ? this.newtilePos : null,
      this.newtile > 0 ? this.newtile : null
    );
    this.animation = new AnimationState({
      appearIndex: animationMetadata.appearTile?.index,
      appearValue: animationMetadata.appearTile?.value,
      direction: animationMetadata.direction,
      validMove: true,
      slideDistances: animationMetadata.slide_distances,
      popPositions: animationMetadata.pop_positions,
      effects: this.queuedMoveEffects.slice(),
      pageEffects: this.queuedPageEffects.slice(),
      followUp: this.queuedFollowUpAnimation,
    });
    this.queuedFollowUpAnimation = null;
    this.queuedMoveEffects = [];
    this.queuedPageEffects = [];
  }

  buildViewState() {
    const cellCount = this.rows * this.cols;
    return {
      hiddenMask: new Array(cellCount).fill(false),
      blockedMask: new Array(cellCount).fill(false),
      smallLabels: new Array(cellCount).fill(''),
      tileOverlays: {},
      tileTextOverride: {},
      tileStyleVariant: {},
      coverSprites: {},
    };
  }

  buildHud() {
    return {
      score: Number(this.score) || 0,
      best: Number(this.maxScore) || 0,
      infoText: this.getInfoText(),
      customPanels: [],
    };
  }

  buildMessages() {
    return this.popMessages();
  }

  buildAnimation() {
    return this.animation.toPayload();
  }

  serializeState() {
    const messages = this.buildMessages();
    let status = this.isOver ? 'game_over' : 'running';
    if (messages.trophy) {
      status = 'trophy';
    }
    return {
      gameId: this.gameId,
      title: this.title,
      difficulty: this.difficulty,
      board: flattenBoard(this.board),
      shape: { rows: this.rows, cols: this.cols },
      score: Number(this.score) || 0,
      best: Number(this.maxScore) || 0,
      status,
      animation: this.buildAnimation(),
      view: this.buildViewState(),
      hud: this.buildHud(),
      powerups: {
        enabled: Boolean(this.definition.supportsPowerups),
        counts: { bomb: 0, glove: 0, twist: 0 },
        activeMode: null,
      },
      interaction: {
        active: false,
        mode: null,
        targetType: null,
        phase: null,
      },
      messages,
    };
  }
}
