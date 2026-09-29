import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { move, VARIANTS, DIRECTIONS, initialState, nextMove, initialHash, eventHash, buildReplay, eventBytes, randomSpawn } from '../src/human/engine.js';
import { PRACTICE_PALETTE, createPracticeMoveReminder, practiceCellValue, practiceBoardHex, parsePracticeHex, nodeTime } from '../src/human/practice.js';
import { humanBoardFrame, paddedBoard } from '../src/human/boardAnimation.js';
import { createTerminalOverlay, TERMINAL_OVERLAY_DELAY_MS } from '../src/human/terminalOverlay.js';
const seed = '00000001000000020000000300000004';
const playerProfileSource = readFileSync(new URL('../src/human/PlayerProfile.vue', import.meta.url), 'utf8');
const humanAppSource = readFileSync(new URL('../src/human/HumanApp.vue', import.meta.url), 'utf8');

test('history delete controls are enabled while no delete request is pending', () => {
  assert.match(playerProfileSource, /deletingId\s*=\s*ref\(null\)/);
  assert.match(playerProfileSource, /finally\s*\{\s*deletingId\.value\s*=\s*null;/);
  assert.doesNotMatch(playerProfileSource, /deletingId\s*=\s*ref\(['"]{2}\)/);
});

test('cached profile releases inactive poster resources and bounds API results', () => {
  assert.match(playerProfileSource, /onDeactivated\(\(\) => deactivateResources\(\{ releasePoster: true \}\)\)/);
  assert.match(playerProfileSource, /posterCanvas\.value\.width = 1; posterCanvas\.value\.height = 1/);
  assert.match(playerProfileSource, /remember\(historyCache, key, result, 12\)/);
  assert.match(playerProfileSource, /remember\(bestCache, key, result, 8\)/);
});

test('background Play tabs do not refresh the game clock or speed display', () => {
  assert.match(humanAppSource, /if \(!document\.hidden && view\.value === 'game' && run\.value\?\.firstMoveAt/);
  assert.match(humanAppSource, /if \(!document\.hidden && view\.value === 'game' && !practice\.value && playSettings\.value\.showSpeed/);
});

test('terminal overlay waits two seconds and stays dismissed for that game', () => {
  let visible = false, scheduled = null, cleared = 0;
  const overlay = createTerminalOverlay({
    setTimer(callback, delay) { scheduled = { callback, delay }; return 7; },
    clearTimer() { cleared += 1; },
    onVisible(value) { visible = value; },
  });
  overlay.update('run-one', true);
  assert.equal(visible, false);
  assert.equal(scheduled.delay, TERMINAL_OVERLAY_DELAY_MS);
  scheduled.callback();
  assert.equal(visible, true);
  overlay.dismiss('run-one');
  assert.equal(visible, false);
  scheduled = null;
  overlay.update('run-one', true);
  assert.equal(scheduled, null);
  overlay.update('run-two', true);
  assert.equal(scheduled.delay, 2000);
  overlay.update('run-two', false);
  assert.equal(visible, false);
  assert.ok(cleared >= 1);
  overlay.dispose();
});

test('rectangular animation frames preserve source positions, merge destinations and spawn', () => {
  for (const [rows, cols] of Object.values(VARIANTS)) {
    const board = Array(rows * cols).fill(0); board[cols - 2] = 2; board[cols - 1] = 2;
    const moved = move(board, rows, cols, 3), target = [...moved.board]; target[target.length - 1] = 4;
    const frame = humanBoardFrame('move', target, rows, cols, { fromBoard: board, toBoard: target, direction: 3 });
    assert.equal(frame.kind, 'move');
    assert.deepEqual(frame.fromBoard, paddedBoard(board, cols));
    assert.equal(frame.metadata.slide_distances[cols - 2], cols - 2);
    assert.equal(frame.metadata.slide_distances[cols - 1], cols - 1);
    assert.equal(frame.metadata.pop_positions[0], 1);
    assert.equal(frame.metadata.appear_tile.index, (rows - 1) * 4 + cols - 1);
    assert.equal(frame.metadata.appear_tile.value, 4);
  }
});

test('board edits discard stale moves and manual spawn uses an appearance-only frame', () => {
  const from = [2,0,0,0,0,0,0,0], to = [2,4,0,0,0,0,0,0];
  const spawn = humanBoardFrame('spawn', to, 2, 4, { fromBoard: from, toBoard: to, spawn: 1 });
  assert.equal(spawn.kind, 'move'); assert.equal(spawn.metadata.direction, undefined);
  assert.deepEqual(spawn.metadata.appear_tile, { index: 1, value: 4 });
  const edit = humanBoardFrame('edit', Array(8).fill(32768), 2, 4, { fromBoard: from, toBoard: to, spawn: 1 });
  assert.equal(edit.kind, 'snapshot'); assert.equal(edit.toBoard[0],32768);
});

test('HJKL uses the trainer physical-key mapping', () => {
  for (const [key, arrow] of [['KeyH','ArrowLeft'],['KeyJ','ArrowDown'],['KeyK','ArrowUp'],['KeyL','ArrowRight']]) assert.equal(DIRECTIONS[key], DIRECTIONS[arrow]);
});
test('trainer palette browsing, paint, cycle, erase and pending-spawn precedence', () => {
  assert.equal(practiceCellValue(8,null,2),8);
  assert.equal(practiceCellValue(8,128,0),128);
  assert.equal(practiceCellValue(8,128,2),16);
  assert.equal(practiceCellValue(8,128,1),4);
  assert.equal(practiceCellValue(0,2,1),32768);
  assert.equal(practiceCellValue(32768,2,2),0);
  assert.equal(practiceCellValue(65536,2,2),65536);
  assert.equal(practiceCellValue(128,0,0),0);
  assert.equal(practiceCellValue(0,128,0,true),2);
  assert.equal(practiceCellValue(0,128,2,true),4);
  assert.equal(practiceCellValue(8,0,0,true),8);
});
test('position codes preserve main-site order and rectangular cell counts', () => {
  const board=[0,4,4,8,32768,0,2,0];
  assert.equal(practiceBoardHex(board),'0223f010');
  assert.deepEqual(parsePracticeHex('0x0223f010',8),board);
  assert.equal(practiceBoardHex([65536,131072]),'gh');
  assert.deepEqual(parsePracticeHex('gh',2),[65536,131072]);
  assert.deepEqual(parsePracticeHex('1',8),[0,0,0,0,0,0,0,2]);
  assert.equal(parsePracticeHex('100000000',8),null);
  assert.equal(parsePracticeHex('hjkl',8),null);
  assert.equal(parsePracticeHex('i',8),null);
  assert.equal(practiceBoardHex([262144,2]),'');
  assert.equal(PRACTICE_PALETTE.at(-1),32768);
  assert.equal(PRACTICE_PALETTE.includes(65536),false);
});
test('practice reminder fires after 40 moves once per source position, even after undo and reset', () => {
  const reminder = createPracticeMoveReminder();
  reminder.start('game:one:10');
  for (let i = 0; i < 40; i++) assert.equal(reminder.moved(), false);
  assert.equal(reminder.moves, 40);
  assert.equal(reminder.moved(), true);
  reminder.restore(40);
  assert.equal(reminder.moved(), false);
  reminder.reset();
  assert.equal(reminder.moves, 0);
  for (let i = 0; i < 41; i++) assert.equal(reminder.moved(), false);
  reminder.start('game:one:11');
  for (let i = 0; i < 40; i++) assert.equal(reminder.moved(), false);
  assert.equal(reminder.moved(), true);
});
test('node times match replay precision and minute/hour rollover', () => {
  assert.equal(nodeTime(437),'0.437');assert.equal(nodeTime(60188),'1:00.188');assert.equal(nodeTime(3601234),'1:00:01.234');
});

test('rectangular rows/cols and one merge per tile', () => {
  const result = move([2,2,2,2,4,0,4,0],2,4,3);
  assert.deepEqual(result.board,[4,4,0,0,8,0,0,0]); assert.equal(result.score,16);
  assert.equal(move([65536,65536,0,0,0,0,0,0],2,4,3).board[0],131072);
});
test('invalid move neither changes source board nor consumes RNG', () => {
  const board=[2,0,0,0,4,0,0,0]; const before=[...board];
  assert.equal(move(board,2,4,3).changed,false); assert.deepEqual(board,before);
});
test('practice randomness never touches a formal run', () => {
  const run=initialState('test','3x4',seed); const before=structuredClone(run);
  const practice=[...run.board]; randomSpawn(practice,()=>.5); assert.deepEqual(run,before);
});
test('seeked replay matches sequential states for every variant', async () => {
  for(const variant of Object.keys(VARIANTS)){
    let state={...initialState('test',variant,seed),variant}; const events=[],states=[structuredClone(state)];
    let hash=await initialHash('test',variant,seed);
    for(let i=0;i<600;i++){
      let next;for(let d=0;d<4;d++){next=nextMove(state,d,i?170:0);if(next)break;}if(!next)break;
      hash=await eventHash(hash,next.event); events.push(next.event);state=next.state;states.push(structuredClone(state));
    }
    const replay=buildReplay({header:{run_id:'test',variant,seed},events});
    for(const step of [0,1,Math.floor(events.length/2),events.length]) assert.deepEqual(replay.seek(step),states[step]);
    assert.equal(eventBytes(events).length,events.length*5); assert.equal(hash.length,64);
  }
});
