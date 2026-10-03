import test from 'node:test';
import assert from 'node:assert/strict';
import {PUBLIC_ROOM_MODES,publicRoomMode,isPublicRoom,isOpenRoom,publicLobbyRoute,legacyMode,creationValid,publicRoomStatus,personalPublicRooms} from '../src/publicRoomModes.js';

test('both old URLs enter the common lobby; timed alias preserves mode selection',()=>{
  for(const path of ['/duels','/duels/','/time-attacks','/time-attacks/'])assert.ok(publicLobbyRoute(path));
  for(const path of ['/rooms/ABC','/events','/practice','/duels/unknown'])assert.equal(publicLobbyRoute(path),false);
  assert.equal(legacyMode('/time-attacks/'),'time_attack');
  assert.equal(legacyMode('/duels'),null);
});
test('cross-mode active rooms remain visible regardless of history filter',()=>{
  const rooms=[{room_kind:'duel',status:'GAME_A_READY'},{room_kind:'time_attack',status:'GAME_A_PLAYING'},
    {room_kind:'duel',status:'FINISHED'},{room_kind:'time_attack',status:'CANCELLED'},{room_kind:'competition',status:'SEATING'}];
  assert.equal(rooms.filter(isPublicRoom).filter(isOpenRoom).length,2);
  assert.equal(rooms.filter(isPublicRoom).filter(r=>!isOpenRoom(r)).length,2);
});
test('mode definitions retain endpoint payloads and independent drafts',()=>{
  const duel=publicRoomMode('duel'), timed=publicRoomMode('time_attack');
  const a=duel.defaults(),b=duel.defaults();a.projects.push('cargo');assert.equal(b.projects.length,0);
  assert.equal(duel.createMethod,'createDuel');assert.equal(timed.createMethod,'createTimeAttack');
  assert.deepEqual(duel.payload(a),{projects:['cargo']});
  assert.deepEqual(timed.payload({...timed.defaults(),target_value:'2044',target_kind:'board_sum'}),{variant:'4x4',target_kind:'board_sum',target_value:2044});
  for(const mode of PUBLIC_ROOM_MODES)assert.ok(mode.title.every(Boolean)&&mode.rules.every(Boolean)&&mode.summary);
});
test('shared validation respects the distinct time and target rules',()=>{
  const duel=publicRoomMode('duel'), timed=publicRoomMode('time_attack');
  assert.equal(creationValid(duel,{projects:[]},30,''),false);
  assert.equal(creationValid(duel,{projects:['a','a']},30,''),false);
  assert.equal(creationValid(duel,{projects:['a']},.5,''),false);
  assert.equal(creationValid(duel,{projects:['a']},30,''),true);
  assert.equal(creationValid(timed,timed.defaults(),.5,''),true);
  for(const minutes of [0,-1,1441,NaN])assert.equal(creationValid(timed,timed.defaults(),minutes,''),false);
  assert.equal(creationValid(timed,{...timed.defaults(),target_value:9},10,''),false);
  assert.equal(creationValid(timed,timed.defaults(),10,'x'),false);
});
test('room cards never expose internal enum status labels',()=>{
  for(const status of ['SEATING','GAME_A_PLAYING','GAME_B_READY','GAME_C_RESULT','FINISHED','CANCELLED','NEW_UNKNOWN']){
    for(const lang of ['zh','en'])assert.notEqual(publicRoomStatus(status,lang),status);
  }
});
test('privileged listings exclude other players rooms without dropping own completed rooms',async()=>{
  const rows=[{room_code:'A',room_kind:'duel'},{room_code:'B',room_kind:'time_attack'},{room_code:'C',room_kind:'competition'}];
  const read=async code=>({status:'FINISHED',me:{seat:code==='B'?{side:'white'}:null,can_close:false}});
  assert.deepEqual((await personalPublicRooms(rows,read)).map(r=>r.room_code),['B']);
});
