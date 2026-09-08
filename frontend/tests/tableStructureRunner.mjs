import { compileTableStructure, matchTableStructure } from '../src/features/gamer/engine/tableStructure.js';
import { maskLargeTiles } from '../src/features/gamer/engine/tableDispatcher.js';

let input = '';
for await (const chunk of process.stdin) input += chunk;
const { metadata, cases } = JSON.parse(input);
const rules = metadata.map(compileTableStructure);
process.stdout.write(JSON.stringify(cases.map(({ board, count, rule }) =>
  matchTableStructure(board, maskLargeTiles(board, count), count, rules[rule]))));
