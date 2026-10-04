import test from 'node:test';
import assert from 'node:assert/strict';
import { rosterAssignments } from '../src/live/content/competitionRosterAssignments.js';

for (const count of [3, 5, 7]) {
  test(`BO${count} keeps every assigned game on both sides`, () => {
    const games = Array.from({ length: count }, (_, i) => ({
      game_key: String.fromCharCode(65 + i), project_key: `project${i}`,
      players: { yellow: { position: i % 3 + 1 }, white: { position: (i + 1) % 3 + 1 } },
    }));
    for (const side of ['yellow', 'white']) {
      const result = rosterAssignments(games, side, key => `${key}（4×4）`);
      assert.equal(Object.values(result).flat().length, count);
      for (const game of games) {
        assert.ok(result[game.players[side].position].includes(`${game.game_key} · ${game.project_key}`));
      }
      assert.deepEqual(result[games[0].players[side].position], games.filter(game => game.players[side].position === games[0].players[side].position).map(game => `${game.game_key} · ${game.project_key}`));
    }
  });
}
test('unassigned games and empty matches have no roster entries', () => {
  assert.deepEqual(rosterAssignments(undefined, 'yellow', String), {});
  assert.deepEqual(rosterAssignments([{ game_key: 'A', players: { white: { position: 1 } } }], 'yellow', String), {});
});
