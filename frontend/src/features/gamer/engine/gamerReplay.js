import { simulateMove } from './classicMove.js';
import { bytesToBase64, crc32, DIRECTION_TO_NEXT_CODE, encodeUleb128, exactBoardCodes, RANKED_RECORD } from './rankedReplayEncoder.js';

const DIRECTIONS = ['up', 'right', 'down', 'left'];
const validBoard = board => Array.isArray(board) && board.length === 16
  && board.every(value => value === 0 || (Number.isSafeInteger(value) && Math.log2(value) % 1 === 0 && value <= 2 ** 31));
const equalBoards = (left, right) => left.every((value, index) => value === right[index]);

// Independent of rank eligibility: one record for each move on the current branch.
export class GamerReplay {
  constructor(board, { partial = false, now = Date.now() } = {}) {
    if (!validBoard(board)) throw new Error('Invalid replay board');
    this.initialBoard = board.slice();
    this.moves = [];
    this.partial = partial;
    this.lastAt = now;
    this.fours = 0;
  }

  append(direction, spawn, now = Date.now()) {
    const code = DIRECTION_TO_NEXT_CODE[direction];
    if (code === undefined || !spawn || !Number.isInteger(spawn.index) || spawn.index < 0 || spawn.index > 15 || ![2,4].includes(spawn.value)) throw new Error('Invalid replay move');
    const four = spawn.value === 4 ? 1 : 0;
    this.moves.push([code, spawn.index, four, Math.max(0, Math.round(now - this.lastAt))]);
    this.fours += four;
    this.lastAt = now;
  }

  undo(now = Date.now()) {
    const move = this.moves.pop();
    if (move) this.fours -= move[2];
    this.lastAt = now;
  }

  statistics() {
    return { moves: this.moves.length, fours: this.fours,
      rate: this.moves.length ? this.fours / this.moves.length : null, partial: this.partial };
  }

  snapshot() { return { version: 1, initialBoard: this.initialBoard, moves: this.moves, partial: this.partial }; }

  static restore(saved, currentBoard, ranked) {
    let input = saved;
    try {
      // Older ranked sessions already contain enough data to recover a complete branch.
      if (!input && ranked?.eligible && !ranked.usedUndo && !ranked.usedSetBoard && ranked.initialTiles?.length === 2) {
        const initialBoard = Array(16).fill(0);
        for (const [index, bit] of ranked.initialTiles) initialBoard[index] = bit ? 4 : 2;
        input = { version: 1, initialBoard, moves: ranked.records?.filter(record => record[0] === RANKED_RECORD.MOVE).map(record => record.slice(1)), partial: false };
      }
      if (input?.version !== 1 || !Array.isArray(input.moves) || input.moves.length > 200000) throw new Error('Invalid replay');
      const replay = new GamerReplay(input.initialBoard, { partial: Boolean(input.partial) });
      let board = replay.initialBoard.slice();
      for (const move of input.moves) {
        if (!Array.isArray(move) || move.length !== 4 || !move.every(Number.isSafeInteger)
          || move[0] < 0 || move[0] > 3 || move[1] < 0 || move[1] > 15 || ![0,1].includes(move[2]) || move[3] < 0) throw new Error('Invalid move');
        const next = simulateMove(board, DIRECTIONS[move[0]]).board;
        if (equalBoards(board,next) || next[move[1]]) throw new Error('Invalid transition');
        next[move[1]] = move[2] ? 4 : 2;
        replay.moves.push(move.slice());
        replay.fours += move[2];
        board = next;
      }
      if (!validBoard(currentBoard) || !equalBoards(board,currentBoard)) throw new Error('Stale replay');
      return replay;
    } catch {
      return new GamerReplay(currentBoard, { partial: true });
    }
  }

  encode() {
    const initial = this.initialBoard.map((value,index) => [index,value]).filter(([,value]) => value);
    const simple = initial.length === 2 && initial.every(([,value]) => value === 2 || value === 4);
    const bytes = [0x52,0x50,0x4c,0x31,0x44,0,simple ? 2 : 0];
    if (simple) for (const [index,value] of initial) bytes.push(index | (value === 4 ? 16 : 0));
    else {
      // Existing RPL1 Checkpoint record: sixteen 5-bit exponents, little-endian.
      const checkpoint = new Uint8Array(10);
      exactBoardCodes(this.initialBoard).forEach((code,index) => {
        for (let bit=0;bit<5;bit++) if (code & (1<<bit)) checkpoint[Math.floor((index*5+bit)/8)] |= 1<<((index*5+bit)%8);
      });
      bytes.push(130,...checkpoint);
    }
    bytes.push(131,2,4,112,111,119,50); // Existing "pow2" ruleset marker.
    for (const [direction,index,four,delta] of this.moves) {
      bytes.push(direction | (index<<2) | (four<<6),...encodeUleb128(delta));
      if (bytes.length > 500*1024-5) throw new Error('replay_too_large');
    }
    bytes.push(132);
    const checksum = crc32(bytes);
    bytes.push(checksum&255,(checksum>>>8)&255,(checksum>>>16)&255,checksum>>>24);
    const text = 'REPLAY_v1RPL_B64_' + bytesToBase64(Uint8Array.from(bytes));
    if (text.length > 500*1024) throw new Error('replay_too_large');
    return text;
  }
}
