import { DIRECTIONS, moveBoardWithSeals, TournamentGame } from './engine.js';
import { nextRandom, seed32, ticketFloat } from './randomStreams.js';

// The heavy tiles remain in their cells and split the affected movement lines.
// Other tiles still use exactly the same 2048 move/merge implementation.
export function moveHeavyBoard(board, project, direction) {
  const vertical = direction === 'up' || direction === 'down';
  const fixed = board.flatMap((value, index) =>
    value >= 1024 || (vertical ? value === 256 : value === 512) ? [index] : []);
  return moveBoardWithSeals(board, project, direction, fixed);
}

function hasHeavyMove(board, project) {
  return DIRECTIONS.some(direction => moveHeavyBoard(board, project, direction).changed);
}

export class PracticeScoreVariantGame extends TournamentGame {
  reset(increment = true) {
    const snapshot = super.reset(increment);
    this.fissionTimers = new Map();
    this.fissionSequence = 0;
    this.fissionRandomState = seed32(`${this.seed}:fission:${this.restartCount}`);
    this.lastFission = null;
    return snapshot;
  }

  moveForRules(board, rules, direction) {
    return this.project.practiceVariant === 'heavy'
      ? moveHeavyBoard(board, rules, direction)
      : super.moveForRules(board, rules, direction);
  }

  settleOutcome() {
    if (this.project.practiceVariant === 'capacity' && this.board.filter(value => value > 0).length > (this.project.tileLimit ?? 12)) {
      this.finished = true;
      this.outcome = 'tile_limit';
      this.finishedAt ??= performance.now();
      return;
    }
    if (this.project.practiceVariant === 'heavy') {
      if (!hasHeavyMove(this.board, this.project)) {
        this.finished = true;
        this.outcome = 'no_moves';
        this.finishedAt ??= performance.now();
      }
      return;
    }
    super.settleOutcome();
  }

  newFissionTimer() {
    this.fissionRandomState = nextRandom(this.fissionRandomState);
    const min = this.project.fissionMinMoves ?? 16;
    const max = this.project.fissionMaxMoves ?? 40;
    return {
      sequence: this.fissionSequence++,
      remaining: min + this.fissionRandomState % (max - min + 1),
    };
  }

  spawnAfterMove(moved) {
    this.lastFission = null;
    if (this.project.practiceVariant !== 'fission') return super.spawnAfterMove(moved);

    const previous = this.fissionTimers || new Map();
    const sources = new Map(moved.movements.map(item => [item.to, item]));
    const merged = new Set(moved.movements.filter(item => item.merged).map(item => item.to));
    const timers = new Map();
    for (let index = 0; index < this.board.length; index += 1) {
      if (this.board[index] < 1024) continue;
      const source = sources.get(index);
      const existing = !merged.has(index) && previous.get(source?.from);
      timers.set(index, existing ? { ...existing, remaining: existing.remaining - 1 } : this.newFissionTimer());
    }
    this.fissionTimers = timers;

    const due = [...timers.entries()]
      .filter(([, timer]) => timer.remaining <= 0)
      .sort((a, b) => a[1].sequence - b[1].sequence || a[0] - b[0])[0];
    if (!due) return super.spawnAfterMove(moved);
    const empty = this.board.flatMap((value, index) => value === 0 ? [index] : []);
    if (!empty.length) return super.spawnAfterMove(moved);

    // A split consumes the turn's spawn ticket, but does not create an extra 2/4.
    const ticket = this.nextSpawnTicket();
    const [from] = due;
    const to = empty[Math.floor(ticketFloat(ticket, 'position') * empty.length)];
    const parent = this.board[from];
    const child = parent / 2;
    this.board[from] = child;
    this.board[to] = child;
    timers.delete(from);
    if (child >= 1024) {
      timers.set(from, this.newFissionTimer());
      timers.set(to, this.newFissionTimer());
    }
    this.lastFission = { index: from, spawnedIndex: to, parent, value: child };
    return null;
  }

  moveTransitionExtras() {
    return this.lastFission ? { fission: this.lastFission } : {};
  }
}
