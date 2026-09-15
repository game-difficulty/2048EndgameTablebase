import test from 'node:test';
import assert from 'node:assert/strict';
import { bagCaption, mergeBags, visibleBags } from '../src/live/luckyBagState.js';

const bag={id:'one',draw_at:280,drawn_at:null,expires_at:460};
test('countdown uses server time, never goes negative, drawn state is explicit',()=>{
  assert.equal(bagCaption(bag,100,'zh'),'3:00');
  assert.equal(bagCaption(bag,279.2,'en'),'0:01');
  assert.equal(bagCaption(bag,300,'zh'),'开奖中');
  assert.equal(bagCaption({...bag,drawn_at:280},281,'zh'),'已开奖');
  assert.equal(bagCaption({...bag,drawn_at:280},281,'en'),'Results');
});
test('public broadcasts keep private membership but invalidate pending results at draw',()=>{
  const old=mergeBags([],[{...bag,joined:true,award:0}],true);
  assert.equal(mergeBags(old,[bag])[0].joined,true);
  const drawn={...bag,drawn_at:280};
  assert.equal(mergeBags(old,[drawn])[0].resultKnown,false);
  const won=mergeBags(old,[{...drawn,joined:true,award:1000,present:true}],true);
  const refreshed=mergeBags(won,[drawn])[0];
  assert.equal(refreshed.resultKnown,true);
  assert.equal(refreshed.award,1000);
  assert.equal(mergeBags([],won)[0].award,undefined);
});
test('overlapping bags survive run changes and completed bags hide after three minutes',()=>{
  const waiting={...bag,id:'two'};
  const drawn={...bag,drawn_at:280};
  assert.equal(visibleBags([waiting,drawn],459).length,2);
  assert.deepEqual(visibleBags([waiting,drawn],460),[waiting]);
});
