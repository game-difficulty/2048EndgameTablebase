export function canResolveLargePair(board) {
  const large = board.filter(value => value >= 32768);
  return large.length === 2 && large[0] === large[1];
}
