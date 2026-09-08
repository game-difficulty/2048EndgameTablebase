const CELL_INDICES = Array.from({ length: 16 }, (_, index) => index);
const rotateLeft = (cells) => CELL_INDICES.map((index) =>
  cells[(index % 4) * 4 + 3 - Math.floor(index / 4)]);
const flipHorizontal = (cells) => CELL_INDICES.map((index) =>
  cells[Math.floor(index / 4) * 4 + 3 - index % 4]);
const rotations = [CELL_INDICES];
for (let turn = 1; turn < 4; turn += 1) rotations.push(rotateLeft(rotations[turn - 1]));
const TRANSFORMS = [...rotations, ...rotations.map(flipHorizontal)];

export function compileTableStructure(metadata) {
  if (metadata?.version !== 1 || metadata.transforms !== 'dihedral8'
    || !Array.isArray(metadata.pattern_masks)
    || metadata.pattern_masks.some((mask) => typeof mask !== 'string' || !/^[0-9a-f]{16}$/i.test(mask))) {
    return null;
  }
  // Compile sparse nibble tests once per catalog, equivalent to native (board & mask) == mask.
  // An empty native mask list accepts any structure.
  return (metadata.pattern_masks.length ? metadata.pattern_masks : ['0000000000000000'])
    .flatMap((mask) => TRANSFORMS.map((indices) => [...mask].flatMap((digit, index) => {
      const bits = parseInt(digit, 16);
      return bits ? [[indices[index], bits]] : [];
    })));
}

export function matchTableStructure(board, maskedBoard, count, rules) {
  if (!rules || !Number.isInteger(count) || count < 0 || count > 16
    || board.length !== 16 || maskedBoard.length !== 16
    || board.some((value) => value !== 0 && (!Number.isInteger(Math.log2(value)) || value > 2 ** 31))
    || maskedBoard.some((value) => value > 32768)) return 'unknown';

  // numpy.argpartition and JS sort may choose different cells at a tied mask boundary.
  // Existing 32K cells are already masked; smaller tied cells must still be queried.
  if (count > 0 && count < 16) {
    const sorted = [...board].sort((a, b) => b - a);
    if (sorted[count - 1] === sorted[count] && sorted[count] !== 32768) return 'unknown';
  }
  const codes = maskedBoard.map((value) => value ? Math.log2(value) : 0);
  return rules.some((tests) => tests.every(([index, bits]) => (codes[index] & bits) === bits))
    ? 'match' : 'mismatch';
}
