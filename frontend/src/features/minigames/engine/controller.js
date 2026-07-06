import { createEngine } from './games/index.js';
import { buildMenuPayload, MINIGAME_BY_ID } from './registry.js';
import {
  activatePowerup,
  applyTargetAction,
  buildInteractionPayload,
  buildPowerupsPayload,
  cancelPowerupInteraction,
  clonePowerupCounts,
  defaultPowerupCounts,
  maybeAwardRandomPowerup,
} from './powerups.js';

export class MinigameController {
  constructor({ difficulty = 1, summaries = {}, snapshotKey } = {}) {
    this.difficulty = Number(difficulty) ? 1 : 0;
    this.currentGameId = '';
    this.engine = null;
    this.powerupCounts = { bomb: 0, glove: 0, twist: 0 };
    this.activeMode = null;
    this.interactionPhase = 0;
    this.selectionCache = null;
    this.summaries = summaries || {};
    this.snapshotKey = snapshotKey;
  }

  setSummaries(summaries) {
    this.summaries = summaries || {};
  }

  setDifficulty(difficulty) {
    this.difficulty = Number(difficulty) ? 1 : 0;
  }

  menuPayload() {
    return {
      ...buildMenuPayload(this.difficulty, this.summaries, this.snapshotKey),
      currentGameId: this.currentGameId,
    };
  }

  async startGame(gameId, snapshot = null) {
    const definition = MINIGAME_BY_ID[String(gameId || '')];
    if (!definition) {
      throw new Error('Unknown minigame');
    }
    if (!definition.implemented) {
      throw new Error(`Minigame '${definition.title}' is not implemented yet.`);
    }
    const engineSnapshot = snapshot?.engine && typeof snapshot.engine === 'object' ? snapshot.engine : null;
    this.engine = createEngine(definition, this.difficulty, engineSnapshot);
    this.currentGameId = definition.id;
    this.activeMode = null;
    this.interactionPhase = 0;
    this.selectionCache = null;
    this.powerupCounts = clonePowerupCounts(snapshot?.powerupCounts, defaultPowerupCounts(this));
    return this.statePayload();
  }

  async newGame() {
    if (!this.engine) {
      throw new Error('No active minigame');
    }
    this.engine.setupNewRound();
    cancelPowerupInteraction(this, { clearAnimation: false });
    this.powerupCounts = defaultPowerupCounts(this);
    return this.statePayload();
  }

  backToMenu() {
    this.resetRuntime();
    return this.menuPayload();
  }

  close() {
    this.resetRuntime();
  }

  resetRuntime() {
    this.currentGameId = '';
    this.engine = null;
    this.powerupCounts = { bomb: 0, glove: 0, twist: 0 };
    this.activeMode = null;
    this.interactionPhase = 0;
    this.selectionCache = null;
  }

  async move(direction) {
    if (!this.engine) {
      throw new Error('No active minigame');
    }
    const previousScore = Number(this.engine.score) || 0;
    await this.engine.doMove(direction);
    const awarded = maybeAwardRandomPowerup(this, (Number(this.engine.score) || 0) - previousScore);
    if (awarded) {
      this.engine.queueMessage('toast', `+1 ${awarded.charAt(0).toUpperCase()}${awarded.slice(1)}`);
    }
    return this.statePayload();
  }

  requestInfo() {
    if (!this.engine) {
      throw new Error('No active minigame');
    }
    this.engine.requestInfo();
    return this.statePayload();
  }

  triggerCustomAction({ key, phase } = {}) {
    if (!this.engine) {
      throw new Error('No active minigame');
    }
    this.engine.handleCustomAction(String(key || ''), String(phase || 'trigger'));
    return this.statePayload();
  }

  usePowerup(mode) {
    if (!this.engine) {
      throw new Error('No active minigame');
    }
    activatePowerup(this, mode);
    return this.statePayload();
  }

  cancelInteraction() {
    cancelPowerupInteraction(this);
    return this.statePayload();
  }

  targetAction(index) {
    if (!this.engine) {
      throw new Error('No active minigame');
    }
    applyTargetAction(this, Number(index));
    return this.statePayload();
  }

  statePayload() {
    if (!this.engine) {
      throw new Error('No active minigame');
    }
    const payload = this.engine.serializeState();
    payload.powerups = buildPowerupsPayload(this);
    payload.interaction = buildInteractionPayload(this);
    payload.snapshot = {
      schemaVersion: 2,
      gameId: this.currentGameId,
      difficulty: this.difficulty,
      engine: this.engine.exportSnapshot(),
      powerupCounts: { ...this.powerupCounts },
    };
    return payload;
  }
}
