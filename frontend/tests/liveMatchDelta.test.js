import test from 'node:test';
import assert from 'node:assert/strict';
import { createMatchDeltaDecoder } from '../src/live/matchDelta.js';

const room={id:'competition-demo',protocol:'competition-match-v1'};
const full={type:'snapshot',room_id:room.id,protocol:room.protocol,stream_epoch:'one',stream_sequence:1,
  match:{match_public_key:'demo',generation:1,current_game:'A',phase:'GAME_A_PLAYING',content_sequence:10,
    teams:{yellow:'Yellow'},project_public_views:{yellow:{sequence:1,payload:{score:4},frames:[]},white:null}}};
const delta={type:'match_delta',room_id:room.id,protocol:room.protocol,stream_epoch:'one',base_sequence:1,stream_sequence:2,
  fields:{set:{server_time:123},unset:[]},match:{set:{content_sequence:11},unset:['teams']},
  views:{set:{yellow:{sequence:2,payload_from_last_frame:true,frames:[{sequence:2,payload:{score:8}}]}},unset:['white']}};

test('delta reconstructs the last board, nulls/deletions and immutable metadata',()=>{
  const codec=createMatchDeltaDecoder(room);codec.decode(full);const state=codec.decode(delta);
  assert.equal(state.type,'snapshot');assert.equal(state.match.content_sequence,11);
  assert.deepEqual(state.match.project_public_views.yellow.payload,{score:8});
  assert.equal(state.match.project_public_views.white,undefined);assert.equal(state.match.teams,undefined);
  assert.equal(full.match.teams.yellow,'Yellow');assert.equal(full.match.project_public_views.yellow.sequence,1);
});

test('missing baseline, skipped, duplicate, foreign epoch and foreign room fail closed',()=>{
  for(const corrupt of [delta,{...delta,base_sequence:0},{...delta,stream_sequence:3},{...delta,stream_epoch:'old'},{...delta,room_id:'wrong'}]){
    const codec=createMatchDeltaDecoder(room);if(corrupt!==delta)codec.decode(full);
    assert.throws(()=>codec.decode(corrupt));
  }
  const codec=createMatchDeltaDecoder(room);codec.decode(full);codec.decode(delta);
  assert.throws(()=>codec.decode(delta));
});

test('a reconnect discards the old baseline; full next-game snapshots resume immediately',()=>{
  const codec=createMatchDeltaDecoder(room);codec.decode(full);codec.reset();assert.throws(()=>codec.decode(delta));
  const next={...full,stream_epoch:'new',stream_sequence:20,match:{...full.match,current_game:'B',phase:'GAME_B_PLAYING'}};
  assert.equal(codec.decode(next).match.current_game,'B');
  assert.throws(()=>codec.decode(delta));
});

test('repeated metadata-only deltas never mutate a queued snapshot or its board',()=>{
  const codec=createMatchDeltaDecoder(room);const first=codec.decode(full);
  const state=codec.decode({...delta,match:{set:{team_clocks:{yellow:5000}},unset:[]},views:{set:{},unset:[]}});
  assert.equal(state.match.project_public_views.yellow.sequence,1);assert.equal(first.match.team_clocks,undefined);
});
