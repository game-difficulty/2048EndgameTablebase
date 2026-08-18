import {
  PERFORMANCE_LABELS,
  evaluationOfPerformance,
} from './replayAnalysis.js';
import {
  REPLAY_DIRECTIONS,
  boardForReplayStep,
  boardHex,
  buildStepTransition,
  decodeBoard,
  decodeReplayChange,
  transitionMatchesNextSnapshot,
} from './replayTransition.js';

export class ReplayController {
  constructor({ replay, analysis, pattern = '', source = '', useVariant = false }) {
    this.replay = replay;
    this.analysis = analysis;
    this.pattern = String(pattern || '');
    this.source = String(source || '');
    this.useVariant = !!useVariant;
    this.currentStep = 0;
  }

  setStep(step, { animate = false, previousStep = this.currentStep } = {}) {
    const target = Math.max(0, Math.min(Number(step) || 0, this.replay.moveCount));
    this.currentStep = target;
    return this.state({ animate, previousStep });
  }

  step(delta) {
    const previousStep = this.currentStep;
    return this.setStep(previousStep + (Number(delta) || 0), {
      animate: Number(delta) === 1,
      previousStep,
    });
  }

  state({ animate = false, previousStep = null } = {}) {
    const step = this.currentStep;
    const total = this.replay.moveCount;
    const boardEncoded = boardForReplayStep(this.replay, step, this.useVariant);
    let metadata = {};
    if (
      animate
      && previousStep != null
      && previousStep + 1 === step
      && previousStep < total
      && transitionMatchesNextSnapshot(this.replay, previousStep, this.useVariant)
    ) {
      metadata = buildStepTransition(this.replay, previousStep, this.useVariant)?.metadata || {};
    }

    const results = {};
    let currentMove = null;
    let bestMove = null;
    let loss = null;
    let goodnessOfFit = null;
    let combo = 0;
    let evaluation = null;
    if (step < total) {
      const offset = step * 4;
      REPLAY_DIRECTIONS.forEach((direction, index) => {
        results[direction] = this.replay.rates[offset + index] / 4e9;
      });
      bestMove = REPLAY_DIRECTIONS.reduce((best, direction) => (
        best == null || results[direction] > results[best] ? direction : best
      ), null);
      currentMove = decodeReplayChange(this.replay.changes[step]).direction;
      if (!this.analysis.forced[step]) {
        loss = this.analysis.losses[step];
        evaluation = evaluationOfPerformance(loss);
      }
      goodnessOfFit = this.analysis.goodnessOfFit[step];
      combo = this.analysis.combo[step];
    }

    return {
      board: decodeBoard(boardEncoded),
      animation: metadata,
      hex_str: boardHex(boardEncoded),
      loaded: true,
      status: this.source ? `Loaded ${this.source}` : '',
      pattern: this.pattern,
      source: this.source,
      current_step: step,
      total_steps: total,
      results,
      current_move: currentMove,
      best_move: bestMove,
      loss,
      goodness_of_fit: goodnessOfFit,
      combo,
      evaluation,
      summary: this.analysis.summary,
      performance_labels: [...PERFORMANCE_LABELS],
    };
  }
}
