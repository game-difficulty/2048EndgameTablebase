// Dihedral symmetries preserve adjacency and the corner-cutoff restriction.
export function orientSeal(index, size, orientation) {
  let row = Math.floor(index / size), col = index % size;
  if (orientation & 4) col = size - 1 - col;
  for (let turn = 0; turn < (orientation & 3); turn += 1) [row, col] = [col, size - 1 - row];
  return row * size + col;
}

export function unorientSeal(index, size, orientation) {
  for (let original = 0; original < size * size; original += 1) {
    if (orientSeal(original, size, orientation) === index) return original;
  }
  throw new Error('Seal cell is outside the board');
}
