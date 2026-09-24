import test from 'node:test';
import assert from 'node:assert/strict';
import { predictionAnnouncement, predictionAnnouncementSegments } from '../src/features/roomActivities/predictionAnnouncement.js';
import { mergeLiveChat } from '../src/features/gifts/giftArtwork.js';
const event = { type:'prediction_reward',id:'reward:1',at:100,player_name:'Lume',tier:'65k',names:['Alice','Bob'],recipient_count:2,total_units:4500000 };
test('reward announcement shows combined credited total and no settlement disclaimer', () => {
  assert.equal(predictionAnnouncement(event),'🎉 Lume 合出 65k！独立加注奖励：Alice、Bob共获得 4,500 Token。');
  assert.equal(predictionAnnouncement({...event,tier:'65k+32k',total_units:25500000}), '🎉 Lume 合出 65k＋32k！独立加注奖励升级：Alice、Bob共获得 25,500 Token。');
});
test('three names at most, count includes all recipients, English supported', () => {
  const many={...event,names:['Alice','Bob','C','hidden'],recipient_count:6};
  assert.match(predictionAnnouncement(many),/Alice、Bob、C…等 6 人共获得 4,500 Token/);
  assert.doesNotMatch(predictionAnnouncement(many),/hidden/);
  assert.match(predictionAnnouncement(many,'en'),/Alice, Bob, C… \(6 viewers\) received 4,500 Token in total/);
  assert.doesNotMatch(predictionAnnouncement({...many,recipient_count:3}),/…/);
});
test('replayed outbox and history events produce one chat row per milestone', () => {
  const upgraded={...event,id:'reward:2',tier:'65k+32k'};
  assert.deepEqual(mergeLiveChat([event],[event,upgraded,event]),[event,upgraded]);
});
test('player, usernames and aggregate amount are separate emphasis segments', () => {
  assert.deepEqual(predictionAnnouncementSegments(event).filter(p=>p.kind),[
    {text:'Lume',kind:'player'},{text:'Alice',kind:'name'},{text:'Bob',kind:'name'},
    {text:'4,500 Token',kind:'amount'},
  ]);
  const name='<img src=x onerror=alert(1)>';
  assert.equal(predictionAnnouncementSegments({...event,names:[name]})[3].text,name);
});

test('token displays round to one decimal without changing stored units', () => {
  for (const lang of ['zh', 'en']) {
    for (const [units, shown] of [[12344, '12.3'], [12350, '12.4'], [1999, '2'], [1000, '1'], [49, '0']]) {
      const reward = { ...event, total_units: units };
      assert.ok(predictionAnnouncement(reward, lang).includes(`${shown} Token`));
      assert.equal(reward.total_units, units);
    }
  }
});
