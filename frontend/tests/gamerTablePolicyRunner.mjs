import { TableDispatcher, maskLargeTiles, packedLookupBoard } from '../src/features/gamer/engine/tableDispatcher.js';
import { planGamerSpawn } from '../src/features/gamer/engine/gamerSpawn.js';
import { Xoshiro128StarStar } from '../src/features/gamer/engine/seededRng.js';
let input = '';
for await (const chunk of process.stdin) input += chunk;
const payload = JSON.parse(input);
const results = payload.cases.map(({ board, probes }) => {
  const dispatcher = new TableDispatcher(payload.tables, 0.1);
  dispatcher.reset(board);
  const candidates = dispatcher.candidates();
  return {
    candidates: candidates.map(({ table, type }) => [table.fullPattern, type]),
    masks: candidates.map(({ table }) => packedLookupBoard(maskLargeTiles(board, table.n))),
    decisions: probes.map(({ table, type, value, dtype }) => {
      dispatcher.cooldowns.clear();
      return dispatcher.accept({ table, type }, { results: { left: value }, dtype });
    }),
  };
});
const spawns = (payload.spawns || []).map(({ board, state, options }) =>
  planGamerSpawn(board, new Xoshiro128StarStar(state), options));
process.stdout.write(JSON.stringify({ results, spawns }));
