import { nextMove, isOver, VARIANTS } from '../../../frontend/src/human/engine.js';

export { VARIANTS };
export function targetReached(config, board) {
  return config.target_kind === 'tile' ? Math.max(...board) >= config.target_value : board.reduce((a,b)=>a+b,0) === config.target_value;
}
export function attemptStatus(config, state) {
  if (targetReached(config, state.board)) return 'reached';
  if (config.target_kind === 'board_sum' && state.board.reduce((a,b)=>a+b,0) > config.target_value) return 'overshot';
  return isOver(state.board, config.variant) ? 'no_moves' : 'playing';
}
export class TimeAttackRuntime {
  constructor(attempt, config) {
    this.id = attempt.id;
    this.config = config;
    // Snapshots can be Vue proxies; the wire state is JSON-only.
    this.state = { ...JSON.parse(JSON.stringify(attempt.state)), variant: config.variant };
    this.status = attempt.status;
    this.pending = [];
  }
  move(direction, elapsed) {
    if (this.status !== 'playing' || this.pending.length >= 128) return false;
    const delta = Math.max(0, Math.floor(elapsed) - this.state.elapsed);
    const next = nextMove(this.state, direction, delta);
    if (!next) return false;
    this.state = next.state;
    this.pending.push(next.event);
    this.status = attemptStatus(this.config, this.state);
    return true;
  }
  batch(commandId) {
    if (!this.pending.length) return null;
    return { action:'submit', attempt_id:this.id, base_sequence:this.state.seq-this.pending.length,
      events:this.pending.slice(0,64), command_id:commandId };
  }
  acknowledge(packet) {
    if (packet.attempt_id !== this.id || packet.base_sequence !== this.state.seq-this.pending.length) throw new Error('stale_ack');
    this.pending.splice(0,packet.events.length);
  }
}
