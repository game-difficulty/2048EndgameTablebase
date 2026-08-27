import { canonicalStateDigest, encodeMgo1, MAX_MGO1_ACTIONS } from '../protocol/index.js';

export const MINIGAME_RANKED_RULES_VERSION = 1;
export const CUSTOM_ACTION_ID = Object.freeze({ peek: 1 });
export const CUSTOM_PHASE_ID = Object.freeze({ trigger: 0, start: 1, end: 2, cancel: 3 });
export const CUSTOM_ACTION_KEY = Object.freeze({ 1: 'peek' });
export const CUSTOM_PHASE_KEY = Object.freeze({ 0: 'trigger', 1: 'start', 2: 'end', 3: 'cancel' });

const clampDelta = (value) => Math.max(0, Math.min(Number.MAX_SAFE_INTEGER, Math.trunc(Number(value) || 0)));

const normalizeOperation = (operation) => {
  if (operation?.type !== 'custom') return { ...operation };
  const actionId = CUSTOM_ACTION_ID[String(operation.key || '').toLowerCase()];
  const phase = CUSTOM_PHASE_ID[String(operation.phase || 'trigger').toLowerCase()];
  if (actionId == null || phase == null) throw new Error('Unsupported ranked minigame custom action.');
  return { type: 'custom', actionId, phase };
};

export function verificationState(controller) {
  return {
    engine: controller.engine?.exportSnapshot?.() || null,
    powerupCounts: { ...(controller.powerupCounts || {}) },
    runtime: controller.runtime?.exportSnapshot?.() || null,
  };
}

export class MinigameRankedRecorder {
  constructor({
    runId,
    userId = null,
    gameId,
    difficulty,
    seedHex,
    rulesVersion = MINIGAME_RANKED_RULES_VERSION,
    runToken = '',
    expiresAt = '',
    submissionState = 'active',
    actions = [],
    startedAtMs = Date.now(),
    lastActionAtMs = null,
    mutableActionCount = 0,
    ended = false,
  }) {
    this.runId = String(runId || '');
    this.userId = userId == null ? null : Number(userId);
    this.gameId = String(gameId || '');
    this.difficulty = Number(difficulty) ? 1 : 0;
    this.seedHex = String(seedHex || '').trim().toLowerCase();
    this.rulesVersion = Math.max(1, Math.trunc(Number(rulesVersion) || 1));
    this.runToken = String(runToken || '');
    this.expiresAt = String(expiresAt || '');
    this.submissionState = String(submissionState || 'active');
    this.actions = Array.isArray(actions) ? actions.map((action) => ({ ...action })) : [];
    this.startedAtMs = Math.max(0, Math.trunc(Number(startedAtMs) || Date.now()));
    this.lastActionAtMs = lastActionAtMs == null
      ? this.startedAtMs
      : Math.max(0, Math.trunc(Number(lastActionAtMs) || this.startedAtMs));
    this.mutableActionCount = Math.max(0, Math.trunc(Number(mutableActionCount) || 0));
    this.ended = Boolean(ended || this.actions.at(-1)?.type === 'end');
  }

  record(operation, atMs, state) {
    if (this.ended) return false;
    if (!operation || typeof operation !== 'object') return false;
    const isEnd = operation.type === 'end';
    const willAddDigest = !isEnd
      && operation.type !== 'digest'
      && (this.mutableActionCount + 1) % 128 === 0;
    const requiredSlots = 1 + (willAddDigest ? 1 : 0) + (isEnd ? 0 : 1);
    if (this.actions.length + requiredSlots > MAX_MGO1_ACTIONS) {
      this.submissionState = 'too_large';
      return false;
    }
    const currentAtMs = Math.max(this.lastActionAtMs, Math.trunc(Number(atMs) || this.lastActionAtMs));
    const action = {
      ...normalizeOperation(operation),
      deltaMs: clampDelta(currentAtMs - this.lastActionAtMs),
    };
    this.actions.push(action);
    this.lastActionAtMs = currentAtMs;
    if (!['digest', 'end'].includes(action.type)) {
      this.mutableActionCount += 1;
      if (this.mutableActionCount % 128 === 0 && state) {
        this.actions.push({
          type: 'digest',
          digest: canonicalStateDigest(state).toString(),
          deltaMs: 0,
        });
      }
    }
    return true;
  }

  finish(atMs, state, reason = 0) {
    if (this.ended) return true;
    const recorded = this.record(
      { type: 'end', reason: Math.max(0, Math.min(255, Number(reason) || 0)) },
      atMs,
      state,
    );
    if (!recorded) return false;
    this.ended = true;
    this.submissionState = 'finished';
    return true;
  }

  encode() {
    if (!this.ended) throw new Error('Ranked minigame record is not finished.');
    return encodeMgo1({
      rulesVersion: this.rulesVersion,
      gameId: this.gameId,
      difficulty: this.difficulty,
      runId: this.runId,
      userId: this.userId,
      seedHex: this.seedHex,
      actions: this.actions,
    });
  }

  get elapsedMs() {
    return this.actions.reduce(
      (total, action) => total + clampDelta(action?.deltaMs),
      0,
    );
  }

  exportSnapshot() {
    return {
      schemaVersion: 1,
      runId: this.runId,
      userId: this.userId,
      gameId: this.gameId,
      difficulty: this.difficulty,
      seedHex: this.seedHex,
      rulesVersion: this.rulesVersion,
      runToken: this.runToken,
      expiresAt: this.expiresAt,
      submissionState: this.submissionState,
      actions: this.actions.map((action) => ({ ...action })),
      startedAtMs: this.startedAtMs,
      lastActionAtMs: this.lastActionAtMs,
      mutableActionCount: this.mutableActionCount,
      ended: this.ended,
    };
  }

  static restore(snapshot) {
    if (!snapshot || Number(snapshot.schemaVersion) !== 1) return null;
    try {
      return new MinigameRankedRecorder(snapshot);
    } catch {
      return null;
    }
  }
}
