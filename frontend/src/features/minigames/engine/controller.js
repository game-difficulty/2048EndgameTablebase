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
import { createMinigameRuntime, restoreMinigameRuntime } from './runtime.js';
import { verificationState } from './rankedRecorder.js';

export class MinigameController {
  constructor({ difficulty = 1, summaries = {}, snapshotKey, runtime = null, onOperation = null } = {}) {
    this.difficulty = Number(difficulty) ? 1 : 0;
    this.currentGameId = '';
    this.engine = null;
    this.powerupCounts = { bomb: 0, glove: 0, twist: 0 };
    this.activeMode = null;
    this.interactionPhase = 0;
    this.selectionCache = null;
    this.summaries = summaries || {};
    this.snapshotKey = snapshotKey;
    this.runtime = runtime || createMinigameRuntime();
    this.onOperation = typeof onOperation === 'function' ? onOperation : null;
    this.lastOperationAccepted = false;
  }

  setRuntime(runtime) {
    this.runtime = runtime || createMinigameRuntime();
  }

  setOperationListener(listener) {
    this.onOperation = typeof listener === 'function' ? listener : null;
  }

  emitOperation(operation) {
    this.onOperation?.({
      operation: { ...operation },
      atMs: this.runtime.now(),
      state: verificationState(this),
    });
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

  async startGame(gameId, snapshot = null, runtime = null) {
    const definition = MINIGAME_BY_ID[String(gameId || '')];
    if (!definition) {
      throw new Error('Unknown minigame');
    }
    if (!definition.implemented) {
      throw new Error(`Minigame '${definition.title}' is not implemented yet.`);
    }
    const engineSnapshot = snapshot?.engine && typeof snapshot.engine === 'object' ? snapshot.engine : null;
    const baseRuntime = runtime || this.runtime;
    this.runtime = runtime || (snapshot?.runtime
      ? restoreMinigameRuntime(snapshot.runtime, {
        clock: baseRuntime?.clock,
        evilSpawn: baseRuntime?.evilSpawn,
      })
      : baseRuntime || createMinigameRuntime());
    this.engine = createEngine(definition, this.difficulty, engineSnapshot, this.runtime);
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
    this.lastOperationAccepted = true;
    const awarded = maybeAwardRandomPowerup(this, (Number(this.engine.score) || 0) - previousScore);
    if (awarded) {
      this.engine.queueMessage('toast', `+1 ${awarded.charAt(0).toUpperCase()}${awarded.slice(1)}`);
    }
    this.emitOperation({ type: 'move', direction: String(direction || '').toLowerCase() });
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
    const actionKey = String(key || '').toLowerCase();
    const actionPhase = String(phase || 'trigger').toLowerCase();
    const changed = this.engine.handleCustomAction(actionKey, actionPhase);
    this.lastOperationAccepted = Boolean(changed);
    if (changed) this.emitOperation({ type: 'custom', key: actionKey, phase: actionPhase });
    return this.statePayload();
  }

  usePowerup(mode) {
    if (!this.engine) {
      throw new Error('No active minigame');
    }
    this.lastOperationAccepted = Boolean(activatePowerup(this, mode));
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
    const mode = this.activeMode;
    const source = Number(this.selectionCache?.sourceIndex ?? -1);
    const target = Number(index);
    const changed = applyTargetAction(this, target);
    this.lastOperationAccepted = Boolean(changed);
    if (changed && mode === 'bomb') this.emitOperation({ type: 'bomb', index: target });
    if (changed && mode === 'twist') this.emitOperation({ type: 'twist', index: target });
    if (changed && mode === 'glove') this.emitOperation({ type: 'glove', source, target });
    return this.statePayload();
  }

  tick() {
    if (!this.engine) throw new Error('No active minigame');
    this.engine.checkGameOver();
    this.lastOperationAccepted = Boolean(this.engine.isOver);
    this.emitOperation({ type: 'tick' });
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
      runtime: this.runtime.exportSnapshot(),
      engine: this.engine.exportSnapshot(),
      powerupCounts: { ...this.powerupCounts },
    };
    return payload;
  }
}
