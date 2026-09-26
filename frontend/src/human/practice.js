// TrainerPage/useTrainerSession semantics, generalized to rectangular human boards.
export const PRACTICE_PALETTE = [0, ...Array.from({ length: 15 }, (_, i) => 2 ** (i + 1))];
export function createPracticeMoveReminder(limit = 40) {
  const warnedOrigins = new Set();
  let origin = null;
  let moves = 0;
  return {
    get moves() { return moves; },
    start(key) { origin = key; moves = 0; },
    restore(count) { moves = count; },
    reset() { moves = 0; },
    moved() {
      moves += 1;
      if (moves <= limit || warnedOrigins.has(origin)) return false;
      warnedOrigins.add(origin);
      return true;
    },
  };
}
export function practiceCellValue(current, selected, button, pending = false) {
  if (pending) return current === 0 ? (button === 2 ? 4 : 2) : current;
  if (selected === null) return current;
  if (button === 0) return selected;
  const index = PRACTICE_PALETTE.indexOf(current);
  if (index < 0) return current;
  return PRACTICE_PALETTE[(index + (button === 2 ? 1 : -1) + PRACTICE_PALETTE.length) % PRACTICE_PALETTE.length];
}
export function practiceBoardHex(board) {
  if (board.some(value => value > 32768)) return '';
  return board.map(value => value ? Math.log2(value).toString(16) : '0').join('');
}
export function parsePracticeHex(text, cells) {
  const hex = String(text).trim().replace(/^0x/i, '').toLowerCase();
  if (!/^[0-9a-f]+$/.test(hex) || hex.length > cells) return null;
  return [...hex.padStart(cells, '0')].map(char => char === '0' ? 0 : 2 ** parseInt(char, 16));
}
export function nodeTime(ms) {
  const value = Math.max(0, Math.round(ms));
  if (value < 60000) return (value / 1000).toFixed(3);
  const hours = Math.floor(value / 3600000), minutes = Math.floor(value / 60000) % 60;
  const seconds = `${Math.floor(value / 1000) % 60}`.padStart(2, '0');
  return `${hours ? hours + ':' + String(minutes).padStart(2, '0') : minutes}:${seconds}.${String(value % 1000).padStart(3, '0')}`;
}
