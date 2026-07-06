import { getEvilCore } from '../../../services/wasm/aiCoreClient.js';
import { flattenBoard } from './utils.js';

let evilCoreModule = null;
let evilGen = null;
let evilCoreUnavailable = false;
let reportedFailure = false;

function encodeExponentBoard(board) {
  let encoded = 0n;
  flattenBoard(board).slice(0, 16).forEach((value, index) => {
    encoded |= BigInt((Number(value) || 0) & 0xf) << BigInt((15 - index) * 4);
  });
  return encoded;
}

export async function generateEvilSpawn(board, depth = 4) {
  if (typeof window === 'undefined') {
    return null;
  }
  if (evilCoreUnavailable) {
    return null;
  }
  try {
    evilCoreModule = evilCoreModule || await getEvilCore();
    if (!evilCoreModule?.EvilGen) {
      return null;
    }
    const encoded = encodeExponentBoard(board);
    evilGen = evilGen || new evilCoreModule.EvilGen(encoded);
    evilGen.reset_board(encoded);
    const result = evilGen.gen_new_num(Math.max(1, Number(depth) || 4));
    const index = Number(result?.[1]);
    const value = Number(result?.[2]);
    if (!Number.isInteger(index) || index < 0 || index >= 16 || !Number.isFinite(value) || value <= 0) {
      return null;
    }
    return { index, value };
  } catch (error) {
    evilCoreUnavailable = true;
    if (!reportedFailure) {
      reportedFailure = true;
      console.error('Minigame EvilGen WASM spawn failed; falling back to random spawn.', error);
    }
    return null;
  }
}

export function disposeEvilGen() {
  if (evilGen?.delete) {
    evilGen.delete();
  }
  evilGen = null;
}
