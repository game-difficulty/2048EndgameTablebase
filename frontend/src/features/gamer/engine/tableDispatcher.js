// Decision policy ported from engine_core/AIPlayer.py:DispatcherCommon.
import { compileTableStructure, matchTableStructure } from './tableStructure.js';

export const TABLE_POLICY_VERSION = 1;

export function maskLargeTiles(board, count) {
  const masked = board.slice();
  const indices = board.map((_, index) => index).sort((a, b) => board[a] - board[b] || a - b);
  for (const index of indices.slice(count ? -count : 0)) masked[index] = 32768;
  if (masked.reduce((sum, value) => sum + value, 0) - count * 32768 < 24) {
    return [32768, 32768, 32768, 32768, 0, 32768, 32768, 0,
      0, 32768, 32768, 0, 32768, 32768, 32768, 32768];
  }
  return masked;
}

export function packedLookupBoard(board) {
  return board.map((value) => value ? Math.min(15, Math.round(Math.log2(value))).toString(16) : '0').join('');
}

export class TableDispatcher {
  constructor(tables = [], spawnRate4 = 0.1) {
    this.cooldowns = new Map();
    this.setTables(tables, spawnRate4);
  }

  setTables(tables, spawnRate4) {
    this.tables = tables.filter((table) => table.ai?.compatible
      && table.ai.policy_version === TABLE_POLICY_VERSION
      && Math.abs(table.spawnRate - spawnRate4) < 0.01
      && !table.pattern.includes('_'))
      .map((table) => ({ ...table, n: table.ai.large_tiles, free: table.ai.free_tiles,
        structureRules: compileTableStructure(table.ai.structure),
        targetExp: Math.log2(Number(table.target)),
        level: table.ai.large_tiles + Math.log2(Number(table.target)) }));
    this.cooldowns.clear();
  }

  reset(board) {
    this.board = board.slice();
    this.counts = new Array(16).fill(0);
    for (const value of board) this.counts[value ? Math.min(15, Math.round(Math.log2(value))) : 0] += 1;
    for (const [key, remaining] of this.cooldowns) {
      if (remaining <= 1) this.cooldowns.delete(key);
      else this.cooldowns.set(key, remaining - 1);
    }
  }

  candidates() {
    const groups = [[], [], []];
    const c = this.counts;
    const largest = [...this.board].sort((a, b) => b - a);
    const readers = (level, n) => this.tables.filter((table) => table.level === level && table.n === n)
      .sort((a, b) => b.free - a.free);
    const route = (table, group) => {
      const freeValid = !table.pattern.includes('free') && !table.fullPattern.includes('free')
        || largest[table.free - 1] === 2 ** table.targetExp;
      groups[freeValid ? group : 2].push(table);
    };
    const unfree = (exp) => {
      for (let j = exp + 1; j < 14; j += 1) {
        if (!c[j]) return c.slice(j).reduce((a, b) => a + b, 0) >= 4;
      }
      return false;
    };
    let largeCount = 0;
    let afterCount = 1;
    const currentLarge = c.slice(9).reduce((a, b) => a + b, 0);
    for (let exp = 15; exp >= 7; exp -= 1) {
      if (c[exp] > 1 && exp !== 15) break;
      if (!c[exp]) afterCount = c.slice(exp + 1).reduce((a, b) => a + b, 0) + 1;
      largeCount += c[exp];
      const level = largeCount + exp;
      if (level < 12 || (level === 12 && exp < 8)) continue;
      if (c[exp]) {
        for (const table of readers(level, largeCount)) {
          const group = exp <= 12 && unfree(exp)
            && ((table.free > 4 && currentLarge > 4) || afterCount > 4) ? 2 : 0;
          route(table, group);
        }
        if (c[exp - 1] < 2) {
          for (const table of readers(level + 1, largeCount)) route(table, afterCount < 5 ? 1 : 2);
          if (level > 12) {
            for (const table of readers(level + 2, largeCount)) if (table.free <= 4) route(table, 2);
          }
        }
      } else {
        for (const table of readers(level, largeCount)) route(table, 2);
      }
    }
    groups[0].sort((a, b) => b.free - a.free);
    groups[1].sort((a, b) => a.n - b.n);
    groups[2].sort((a, b) => a.n - b.n);
    return groups.flatMap((group, index) => group.map((table) => ({ table, type: index + 1 })))
      .filter(({ table }) => !this.cooldowns.has(table.fullPattern));
  }

  accept({ table, type }, payload) {
    const [move, raw] = Object.entries(payload.results || {})[0] || [];
    if (typeof raw !== 'number' || !Number.isFinite(raw)) return null;
    const success = raw + (String(payload.dtype).startsWith('1-') ? 1 : 0);
    const target = Number(table.target);
    const remainder = type === 1
      ? this.board.filter((value) => value < target).reduce((a, b) => a + b, 0)
      : this.board.reduce((a, b) => a + b, 0) % target;
    const halves = this.board.filter((value) => value === target / 2).length;
    const quarters = this.board.filter((value) => value === target / 4).length;
    const certain = success > 0.9999999;
    if ((type === 1 && certain && remainder < 24)
      || (type === 2 && (remainder < 32 || halves > 1))
      || (type === 3 && certain && (remainder > target - 4 || remainder < 24))
      || (certain && (halves >= 2 || (halves >= 1 && quarters >= 2)))) {
      this.cooldowns.set(table.fullPattern, 20);
      return 'AI';
    }
    return success > 0 ? move : null;
  }

  async choose(lookup, isCurrent = () => true) {
    for (const candidate of this.candidates()) {
      const masked = maskLargeTiles(this.board, candidate.table.n);
      if (matchTableStructure(this.board, masked, candidate.table.n, candidate.table.structureRules) === 'mismatch') continue;
      const payload = await lookup(candidate, packedLookupBoard(masked));
      if (!isCurrent()) return null;
      const result = this.accept(candidate, payload);
      if (result) return result;
    }
    return 'AI';
  }
}
