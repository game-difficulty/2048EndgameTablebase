import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ServerClock } from '../../competition/shared/serverClock.mjs';
import { projectionIsOlder, receivedProjectView } from '../../competition/shared/projectStateOrder.mjs';

// Exercise the renderer's actual receive, countdown, team-clock and resume code.
const source=readFileSync(new URL('../src/live/content/CompetitionMatchContent.vue',import.meta.url),'utf8');
function renderer() {
  let local=0,wall=900000;
  class Clock extends ServerClock {constructor(){super({now:()=>local,wall:()=>wall});}}
  const line=prefix=>source.split('\n').find(line=>line.startsWith(prefix));
  const code=`
    let serverClock=new ServerClock();
    const match={value:null},now={value:serverClock.now()},finishPlaybackUntil={value:0},playbackPending={value:{}};
    const sides=['yellow','white'];
    const computed=fn=>({get value(){return fn();}});
    ${line('const currentServerNow=')}
    ${line('const phaseCountdown=')}
    ${line('const clock=')}
    ${source.slice(source.indexOf('function receive(data){'),source.indexOf('let resyncPending='))}
    ${line('function resume()')}
    return {receive,resume,clock,countdown:()=>phaseCountdown.value,state:()=>match.value,tick:()=>{now.value=serverClock.now();}};
  `;
  const view=new Function('ServerClock','projectionIsOlder','receivedProjectView',code)(Clock,projectionIsOlder,receivedProjectView);
  return {...view,advance:ms=>{local+=ms;view.tick();},setWall:ms=>{wall=ms;}};
}
const iso=ms=>new Date(ms).toISOString();
function packet(time=200000,overrides={},envelope=999999) {
  return {type:'snapshot',server_time:envelope,match:{match_public_key:'match',generation:1,content_sequence:10,
    phase:'LINEUP',server_time:iso(time),phase_timing:{deadline_at:iso(260000)},
    team_clocks:{yellow:{state:'running',remaining_ms:260000-time}},project_public_views:{},...overrides}};
}

test('delayed same-version samples cannot rewind countdown from 48 to 55',()=>{
  const view=renderer();view.receive(packet());view.advance(12000);
  assert.equal(view.countdown(),'00:48');assert.equal(view.clock('yellow'),'00:48');
  view.receive(packet(205000));
  assert.equal(view.countdown(),'00:48');assert.equal(view.clock('yellow'),'00:48');
  view.advance(1000);assert.equal(view.countdown(),'00:47');
});

test('same-version old or invalid source times are rejected, irrespective of envelope time',()=>{
  const view=renderer();view.receive(packet(205000));view.advance(1000);
  view.receive(packet(200000,{phase:'FIRST_PICK_BAN'},999999999));
  assert.equal(view.state().phase,'LINEUP');assert.equal(view.countdown(),'00:54');
  view.receive(packet(200000,{server_time:null,phase:'FIRST_PICK_BAN'}));
  assert.equal(view.state().phase,'LINEUP');
});

test('duplicate snapshot repairs missing frames without stopping the display clock',()=>{
  const view=renderer();
  const frame=sequence=>({sequence,payload:{board:[[sequence]]}});
  const project=frames=>({yellow:{generation:1,sequence:3,payload:{board:[[3]]},frames}});
  view.receive(packet(200000,{project_public_views:project([frame(3)])}));view.advance(12000);
  view.receive(packet(200000,{project_public_views:project([frame(1),frame(2),frame(3)])}));
  assert.deepEqual(view.state().project_public_views.yellow.frames.map(f=>f.sequence),[1,2,3]);
  assert.equal(view.countdown(),'00:48');
});

test('newer revisions accept official extra time and changed deadlines',()=>{
  const view=renderer();view.receive(packet());view.advance(12000);
  view.receive(packet(212000,{content_sequence:11,phase_timing:{deadline_at:iso(280000)},
    team_clocks:{yellow:{state:'stopped',remaining_ms:78000}}}));
  assert.equal(view.countdown(),'01:08');assert.equal(view.clock('yellow'),'01:18');
  view.advance(1000);assert.equal(view.clock('yellow'),'01:18');
  view.receive(packet(213000,{content_sequence:12,team_clocks:{yellow:{state:'running',remaining_ms:78000}}}));
  view.advance(1000);assert.equal(view.clock('yellow'),'01:17');
});

test('generation and room changes reset the source clock but old generations never replace them',()=>{
  const view=renderer();view.receive(packet());view.advance(12000);
  view.receive(packet(100000,{generation:2,content_sequence:1,phase_timing:{deadline_at:iso(160000)}}));
  assert.equal(view.countdown(),'01:00');
  view.receive(packet(999000,{generation:1,content_sequence:999}));
  assert.equal(view.state().generation,2);assert.equal(view.countdown(),'01:00');
  view.receive(packet(50000,{match_public_key:'new',generation:1,content_sequence:1,phase_timing:{deadline_at:iso(110000)}}));
  assert.equal(view.countdown(),'01:00');
});

test('device and live host wall-clock changes cannot affect competition time, including resume',()=>{
  const view=renderer();view.receive(packet(200000,{},1));
  assert.equal(view.countdown(),'01:00');
  view.setWall(-99999999);view.advance(12000);view.resume();
  assert.equal(view.countdown(),'00:48');
  view.receive(packet(205000,{},999999999));view.advance(1000);
  assert.equal(view.countdown(),'00:47');
});

test('invalid initial time never imports the unrelated live host clock',()=>{
  const view=renderer();view.receive(packet(200000,{server_time:null},999999999));
  view.receive(packet());
  assert.equal(view.countdown(),'01:00');
});

test('new viewer uses the relayed source clock for an idle cached projection',()=>{
  const view=renderer();
  view.receive({...packet(),competition_server_time:212});
  assert.equal(view.countdown(),'00:48');assert.equal(view.clock('yellow'),'00:48');
  view.advance(1000);
  view.receive({...packet(),competition_server_time:205});
  assert.equal(view.countdown(),'00:47');assert.equal(view.clock('yellow'),'00:47');
  view.receive({...packet(),competition_server_time:null});
  assert.equal(view.countdown(),'00:47');
});
