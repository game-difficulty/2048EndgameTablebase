import { TournamentGame, hasMove, hasMoveWithSeals } from './engine.js';
import { CargoGame } from './cargoEngine.js';
import { PolyominoGame } from './polyominoEngine.js';
import { PracticeSpecialGame } from './practiceSpecialEngine.js';
import { PracticeScoreVariantGame } from './practiceScoreVariants.js';
import { AftershockGame, LookBackGame } from './geometryVariants.js';
import { PROJECT_BY_ID } from './catalog.js';

const STATE_KEYS = ['board', 'tiles', 'cargo', 'score', 'moves', 'revision', 'randomState',
  'shapeState', 'chemicalColorState', 'sealState', 'nextCargoId', 'nextTileId', 'restartCount', 'rows', 'cols',
  'dice', 'wallIndex', 'sealedCells', 'sealRound', 'finished', 'outcome',
  'fissionSequence', 'fissionRandomState', 'originRow', 'originCol', 'quakeState', 'lookBackState'];
const clone = value => JSON.parse(JSON.stringify(value));

export function matchProject(bootstrap) {
  const source = PROJECT_BY_ID[bootstrap.project_ref];
  if (!source && bootstrap.project_ref !== 'standard-2048-test') throw new Error('本客户端尚不支持此比赛项目，请刷新页面。');
  const project = { ...(source || { id: 'standard-2048-test', rows: 4, cols: 4, spawn4Rate: .1 }), strictEvilSpawn: true };
  // Respect the geometry frozen in older room pools as well as new v3 rooms.
  if (project.polyomino && bootstrap.rules_version !== 'tournament-v3') project.cols = 4;
  if (project.shapeShifter && bootstrap.rules_version !== 'tournament-v3') project.shapeGenerationSize = 7;
  if (project.practiceVariant === 'fission' && bootstrap.rules_version === 'tournament-v4') project.resultMetric = 'score';
  if (project.geometryVariant === 'lookback' && bootstrap.rules_version === 'tournament-v5') {
    project.race = false;
    project.allowRestart = false;
    delete project.targetTile;
    delete project.targetCount;
  }
  if (bootstrap.target_tile) project.targetTile = bootstrap.target_tile, project.targetCount = 1;
  return project;
}

export class MatchRuntime {
  constructor(bootstrap, { elapsedMs = 0, running = true, now = () => performance.now(), evilSpawn } = {}) {
    this.bootstrap = bootstrap;
    this.project = matchProject(bootstrap);
    this.now = now;
    this.baseElapsed = elapsedMs;
    this.clockAt = now();
    this.running = running;
    this.frozenElapsed = null;
    const options = { seed: bootstrap.seed, side: bootstrap.side, ...(evilSpawn ? { evilSpawn } : {}) };
    this.game = this.project.cargoTransport ? new CargoGame(this.project, options)
      : this.project.polyomino ? new PolyominoGame(this.project, options)
        : this.project.specialRule ? new PracticeSpecialGame(this.project, options)
          : this.project.geometryVariant === 'aftershock' ? new AftershockGame(this.project, options)
            : this.project.geometryVariant === 'lookback' ? new LookBackGame(this.project, options)
          : this.project.practiceVariant ? new PracticeScoreVariantGame(this.project, options)
            : new TournamentGame(this.project, options);
    this.sequence = Number(bootstrap.sequence || 0);
    this.metricHistory = [];
    if (bootstrap.checkpoint) this.restore(bootstrap.checkpoint);
    if (!this.project.race && !this.metricHistory.length) {
      const initial = this.game.snapshot();
      this.metricHistory.push([0, this.project.resultMetric === 'boardSum' ? initial.boardSum : initial.score]);
    }
    this.game.elapsed = () => this.elapsed();
    this.alignClock();
  }

  elapsed() {
    return this.frozenElapsed ?? Math.min(this.budget(), Math.max(0, this.baseElapsed + (this.running ? this.now() - this.clockAt : 0)));
  }
  budget() {
    return this.bootstrap.team_remaining_at_start_ms ?? Infinity;
  }
  playable() { return this.running && !this.completed() && this.elapsed() < this.budget(); }
  setClock(elapsed, running) {
    if (this.frozenElapsed != null) return;
    // Ordinary snapshots must never rewind a running local stopwatch.
    this.baseElapsed = running ? Math.max(this.elapsed(), elapsed) : elapsed;
    this.clockAt = this.now();
    this.running = running;
    this.alignClock();
  }
  alignClock() { this.game.startedAt = performance.now() - this.elapsed(); }

  checkpoint() {
    const state = {};
    for (const key of STATE_KEYS) if (this.game[key] !== undefined) state[key] = clone(this.game[key]);
    if (this.game.fissionTimers instanceof Map) state.fissionTimers = [...this.game.fissionTimers];
    if (this.project.geometryVariant === 'lookback') state.lookBackHistory = this.game.history.map(item => ({ ...item, board: item.board.slice() }));
    // This project's undo only needs board/score/moves; compact tuples avoid
    // repeatedly transmitting object keys and inapplicable wall metadata.
    if (this.project.allowUndo) state.undo = this.game.history.map(item => [item.score, item.moves, ...item.board]);
    return { version: 1, state, elapsed_ms: this.elapsed(), metric_history:clone(this.metricHistory) };
  }
  restore(checkpoint) {
    if (checkpoint?.version !== 1 || !checkpoint.state) throw new Error('对局恢复数据版本不兼容，请刷新客户端。');
    this.metricHistory = clone(checkpoint.metric_history || []);
    for (const key of STATE_KEYS) if (checkpoint.state[key] !== undefined) this.game[key] = clone(checkpoint.state[key]);
    if (this.game.fissionTimers instanceof Map) this.game.fissionTimers = new Map(checkpoint.state.fissionTimers || []);
    if (this.project.geometryVariant === 'lookback') this.game.history = clone(checkpoint.state.lookBackHistory || []);
    if (this.project.allowUndo) this.game.history = (checkpoint.state.undo || []).map(([score, moves, ...board]) =>
      ({ score, moves, board, dice: null, wallIndex: null, sealedCells: [], sealRound: 0 }));
    this.game.transition = { kind: 'restore' };
    this.frozenElapsed = this.completed() ? checkpoint.elapsed_ms : null;
  }

  completed() { return this.game.finished && !(this.project.allowRestart && this.game.outcome === 'no_moves'); }

  accept() {
    this.sequence += 1;
    const snap=this.game.snapshot();
    const metric=this.project.resultMetric==='boardSum' ? snap.boardSum : snap.score;
    if(!this.project.race && this.metricHistory.at(-1)?.[1]!==metric)this.metricHistory.push([Math.floor(this.elapsed()),metric]);
    if (this.completed() && this.frozenElapsed == null) {
      this.frozenElapsed = this.elapsed();
    }
    return this.packet();
  }
  move(direction) {
    if (this.game.finished || !this.playable()) return null;
    this.alignClock();
    const before = this.project.evilSpawn ? this.checkpoint() : null;
    const revision = this.game.revision;
    const result = this.game.move(direction);
    const finish = value => value.changed || this.game.revision !== revision ? this.accept() : null;
    if (!result?.then) return finish(result);
    return result.then(finish, error => { this.restore(before); throw error; });
  }
  action(action) {
    if (!this.playable()) return null;
    if (action === 'surrender') {
      this.game.finished = true;
      this.game.outcome = 'surrendered';
      return this.accept();
    }
    if (action === 'undo' && this.project.allowUndo && this.game.undo()) return this.accept();
    if (action === 'restart' && this.project.allowRestart) {
      this.game.reset(true);
      this.alignClock();
      return this.accept();
    }
    return null;
  }
  tick() {
    return null;
  }
  stopRace() {
    if (this.completed()) return null;
    this.game.finished = true;
    this.game.outcome = 'opponent_finished';
    return this.accept();
  }
  packet() {
    const snapshot = this.game.snapshot();
    const project = this.project;
    const cols = snapshot.cols, rows = snapshot.rows;
    const board = Array.from({ length: rows }, (_, r) => snapshot.board.slice(r * cols, (r + 1) * cols));
    const payload = {
      ...snapshot, board, move_count: snapshot.moves, elapsed_ms: this.elapsed(),
      finished: this.completed(), outcome: this.completed() ? snapshot.outcome : null,
      board_sum: snapshot.boardSum, last_transition: snapshot.transition,
      result_metric: project.resultMetric === 'boardSum' ? 'board_sum' : project.race ? 'race' : 'score',
      allow_restart: Boolean(project.allowRestart), allow_undo: Boolean(project.allowUndo),
      can_undo: Boolean(snapshot.canUndo), evil_spawn: Boolean(project.evilSpawn),
      mirror_portals: Boolean(project.mirrorPortals), shape_shifter: Boolean(project.shapeShifter || project.geometryVariant === 'aftershock'),
      aftershock: project.geometryVariant === 'aftershock',
      isolated_island: Boolean(project.isolatedIsland), wall_index: snapshot.wallIndex,
      target_sum: project.targetSum, target_tile: project.targetTile, target_count: project.targetCount,
      current_target_count: snapshot.targetCount, sealed_cells: snapshot.sealedCells,
      next_seal_in: snapshot.nextSealIn, time_limit_ms: project.timeLimitMs,
      no_moves: project.allowUndo || project.allowRestart ? !(project.sealEveryMoves
        ? hasMoveWithSeals(snapshot.board, { ...project, rows, cols }, snapshot.sealedCells)
        : hasMove(snapshot.board, { ...project, rows, cols })) : snapshot.finished,
    };
    delete payload.transition;
    const value = project.resultMetric === 'boardSum' ? snapshot.boardSum : snapshot.score;
    return { instance_id: this.bootstrap.instance_id, sequence: this.sequence,
      payload, checkpoint: this.checkpoint(), result_value: value,
      elapsed_ms: this.elapsed(), finished: this.completed(), outcome: this.completed() ? snapshot.outcome : null };
  }
}

// One request in flight; checkpoints coalesce, but every public frame survives
// until acknowledged. A long outage beyond the bounded window uses a snapshot.
export class LatestStateSender {
  constructor({ send, onAck = () => {}, onError = () => {}, interval = 100 }) {
    Object.assign(this, { send, onAck, onError, interval });
    this.pending = null; this.inflight = false; this.timer = null; this.closed = false;
    this.frames = [];
  }
  push(packet) {
    if (this.closed) return;
    if (!this.pending || packet.sequence >= this.pending.sequence) this.pending = packet;
    if (!this.frames.length || packet.sequence > this.frames.at(-1).sequence) {
      this.frames.push({ sequence: packet.sequence, payload: packet.payload });
      if (this.frames.length > 128) this.frames.shift();
    }
    if (!this.inflight && !this.timer) this.timer = setTimeout(() => this.flush(), packet.finished ? 0 : this.interval);
  }
  async flush() {
    clearTimeout(this.timer); this.timer = null;
    if (this.closed || this.inflight || !this.pending) return;
    const packet = this.pending; this.pending = null; this.inflight = true;
    let delay = this.interval;
    try {
      const frames = this.frames.filter(frame => frame.sequence <= packet.sequence);
      // Leave room for the current checkpoint within the server's 1 MiB limit.
      while (frames.length > 1 && JSON.stringify({ ...packet, frames }).length > 750000) frames.shift();
      const ack = await this.send({ ...packet, frames });
      this.frames = this.frames.filter(frame => frame.sequence > (ack?.accepted_sequence ?? packet.sequence));
      if (!this.closed) this.onAck(ack, packet);
    } catch (error) {
      if (!this.closed) {
        const retry = this.onError(error) !== false;
        if (retry && (!this.pending || packet.sequence > this.pending.sequence)) this.pending = packet;
        if (!retry) this.pending = null;
        delay = 1000;
      }
    } finally {
      this.inflight = false;
      if (!this.closed && this.pending) this.timer = setTimeout(() => this.flush(), this.pending.finished && delay < 1000 ? 0 : delay);
    }
  }
  close() { this.closed = true; clearTimeout(this.timer); this.pending = null; this.frames = []; }
}
