import { ALL_PROJECTS } from './projects/catalog.js';
const english = [
 ['True Huarong Dao','Use the first 10 valid moves to organize the board. Special pieces then enter from above, slide as whole units, and leave through the bottom-center exit. Play ends when no legal moves remain. Deliver as many pieces as possible.'],
 ['50% Fours','Each new tile has a 50% chance of being a 4. One run decides the result. Compare scores when both players have no legal moves remaining.'],
 ['Every Step a Struggle','EvilGen chooses a difficult spawn position on every move. One run decides the result. Compare scores when both players have no legal moves remaining.'],
 ['Pure-2 Full Board Race','Only 2s spawn. Restarts are allowed. Be the first to reach a tile sum of at least 1022.'],
 ['Extreme Speedrun','Standard spawning. Restarts and undo are allowed. Be the first to reach a tile sum of exactly 2044.'],
 ['Dice Obstacles','Roll at the start: 1–3 places a wall in a corner, 4–5 on an edge, and 6 in the center. Compare tile sums when both players have no legal moves remaining.'],
 ['Mirror Realm','The central cross forms walls. The outer edges are portals: left connects to right and top to bottom. The 64 tiles cannot merge. Be the first to hold ten 64 tiles at once.'],
 ['256 Bricks','A 5×5 board where 256 tiles cannot merge further. One run decides the result. Compare scores when both players have no legal moves remaining.'],
 ['Isolated Island','Island tiles merge only with their own kind and award no points. Compare scores when both players have no legal moves remaining.'],
 ['Shape Shifter','Each run uses a randomly shaped 12-cell board. Tiles cannot move outside it. Compare scores when both players have no legal moves remaining.'],
 ['Hundred-Move Lockdown','Three cells are sealed before the initial tiles spawn. Seals rotate every 100 valid moves. Sealed numbers are preserved and no tiles spawn in sealed cells. Compare scores when no legal moves remain.'],
 ['Getting Bigger','Two 64s form a two-cell 128. Two 128s merge when their cells overlap during movement, forming a two- or three-cell 256. Multi-cell tiles move as a whole; 256s cannot merge. Compare scores when no legal moves remain.'],
 ['Better in Pairs','Special tiles occasionally appear. Two adjacent special tiles bond into a two-cell piece. A new two-cell piece removes the old one. Compare scores when no legal moves remain.'],
 ['Chemical Reaction','Special tiles of two colors occasionally appear. Matching colors disappear on collision; different colors combine into a one-cell wall. Compare scores when no legal moves remain.'],
 ['Timed Bomb','Bombs occasionally spawn with a countdown of 12–32. Each movement reduces it by one; at zero, the bomb becomes a wall. Colliding bombs combine their countdowns. Compare scores when no legal moves remain.'],
 ['Full Load','The board can hold at most 12 number tiles. Exceeding the limit ends the game immediately. Results are ranked by score.'],
 ['Getting Heavier','The 256 tiles move only horizontally, 512s only vertically, and 1024s cannot move. Compare scores when no legal moves remain.'],
 ['Fission','Tiles of 1024 or higher split into two after several moves, replacing that move’s spawn. Compare tile sums when no legal moves remain.'],
 ['Aftershock','A move that creates a tile of 256 or higher triggers one aftershock: one of the original four rows or four columns shifts one cell in a random direction. Compare scores when no legal moves remain.'],
 ['Look Back','A valid move may undo the previous move and spawn two tiles on the restored board. Restarts are allowed. Be the first to make 2048.'],
];
export const englishProjectNames=Object.fromEntries(ALL_PROJECTS.map(project=>[project.id,english[project.order-1][0]]));
export const projectMessages=Object.fromEntries(ALL_PROJECTS.flatMap(project=>{
 const [name,description]=english[project.order-1];
 const dimensions=project.title.match(/（([^）]+)）$/)?.[1];
 return [[project.shortTitle,name],[project.title,dimensions?`${name} (${dimensions})`:name],[project.description,description]];
}));
