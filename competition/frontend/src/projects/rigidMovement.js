// A swipe is one transaction. Internal one-cell rounds are not animation frames.
// Every object is present in the occupancy map, including objects not yet moved.
export const MOVE_VECTORS = { up: [-1, 0], right: [0, 1], down: [1, 0], left: [0, -1] };

export function shiftCells(cells, direction, rows, cols) {
  const [dr, dc] = MOVE_VECTORS[direction];
  const next = cells.map(cell => [Math.floor(cell / cols) + dr, ((cell % cols) + cols) % cols + dc]);
  if (next.some(([r, c]) => r < 0 || r >= rows || c < 0 || c >= cols)) return null;
  return next.map(([r, c]) => r * cols + c);
}

export function settleRigidTiles(input, direction, { cols, step, merge }) {
  const tiles = new Map(input.map(tile => [tile.id, { ...tile, cells: tile.cells.slice() }]));
  const movements = new Map(input.map(tile => [tile.id, { id: tile.id, from: tile.cells.slice(), to: tile.cells.slice() }]));
  const members = new Map(input.map(tile => [tile.id, [tile.id]]));
  const merges = [], removals = [];
  let changed = false, score = 0;
  const [dr, dc] = MOVE_VECTORS[direction];
  const delta = dr * cols + dc;
  // Geometry, never allocation ID, resolves simultaneous collision priority.
  const rank = tile => tile.cells.map(cell => [
    -(Math.floor(cell / cols) * dr + (((cell % cols) + cols) % cols) * dc),
    dr ? ((cell % cols) + cols) % cols : Math.floor(cell / cols),
  ]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).flat();
  const compare = (a, b) => {
    const x = rank(a), y = rank(b);
    for (let i = 0; i < Math.min(x.length, y.length); i++) if (x[i] !== y[i]) return x[i] - y[i];
    return x.length - y.length;
  };
  while (true) {
    const occupied = new Map();
    for (const tile of tiles.values()) for (const cell of tile.cells) occupied.set(cell, tile);
    const candidates = new Map([...tiles.values()].map(tile => [tile.id, step(tile)]));
    const movable = new Set([...tiles.keys()].filter(id => candidates.get(id)));
    // Greatest fixed point: a group can follow itself into vacated cells; an
    // external wall/boundary blocks the whole dependent chain, not just one row.
    let removed;
    do {
      removed = false;
      for (const id of movable) {
        if (candidates.get(id).cells.some(cell => {
          const blocker = occupied.get(cell);
          return blocker && blocker.id !== id && !movable.has(blocker.id);
        })) { movable.delete(id); removed = true; }
      }
    } while (removed);
    if (movable.size) {
      for (const id of movable) {
        const tile = tiles.get(id);
        Object.assign(tile, candidates.get(id));
        for (const source of members.get(id)) movements.get(source).to = movements.get(source).to.map(cell => cell + delta);
      }
      changed = true;
      continue;
    }
    // Only collide after free translations settle. Resolve front collisions
    // first and rebuild dependencies after each merge/disappearance.
    let collision = false;
    for (const tile of [...tiles.values()].sort(compare)) {
      const candidate = candidates.get(tile.id);
      if (!candidate || tile.merged) continue;
      const blockers = [...new Set(candidate.cells.map(cell => occupied.get(cell)).filter(other => other && other.id !== tile.id))];
      if (blockers.length !== 1 || blockers[0].merged) continue;
      const target = blockers[0];
      const reaction = merge?.(tile, target, candidate.cells);
      if (!reaction) continue;
      if (reaction.tile?.cells.some(cell => {
        const other = occupied.get(cell);
        return other && other.id !== tile.id && other.id !== target.id;
      })) continue;
      const sources = [...members.get(target.id), ...members.get(tile.id)];
      const distance = (reaction.to || candidate.cells)[0] - tile.cells[0];
      for (const source of members.get(tile.id)) movements.get(source).to = movements.get(source).to.map(cell => cell + distance);
      tiles.delete(tile.id); tiles.delete(target.id);
      const result = reaction.tile;
      if (result) {
        result.merged = true;
        tiles.set(result.id, result);
        members.set(result.id, sources);
        for (const source of sources) movements.get(source).mergeInto = result.id;
        merges.push({ tile: result, sources });
      } else removals.push(...sources);
      score += reaction.score || 0;
      changed = collision = true;
      break;
    }
    if (!collision) break;
  }
  return { changed, tiles: [...tiles.values()], score, movements: [...movements.values()], merges, removals };
}
