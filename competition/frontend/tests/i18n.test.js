import test from 'node:test';
import assert from 'node:assert/strict';
import { language,t,setLanguage,syncAccountLanguage,languageOverride } from '../src/i18n.js';
import { ALL_PROJECTS,competitionProjectInput } from '../src/projects/catalog.js';
import { ERROR_MESSAGES } from '../src/errorMessages.js';
test('all registered game names and rules have English translations without changing submission data',()=>{
 const before=ALL_PROJECTS.map(competitionProjectInput);
 language.value='en';
 for(const game of ALL_PROJECTS){
  for(const value of [game.title,game.shortTitle,game.description]){
   assert.notEqual(t(value),value,value);assert.doesNotMatch(t(value),/[\u3400-\u9fff]/);
  }
 }
 assert.deepEqual(ALL_PROJECTS.map(competitionProjectInput),before);
 language.value='zh';assert.equal(t('得分'),'得分');
});
test('all mapped action errors are translated',()=>{
 language.value='en';
 for(const message of Object.values(ERROR_MESSAGES))assert.doesNotMatch(t(message),/[\u3400-\u9fff]/,message);
 language.value='zh';
});
test('dynamic confirmations retain participant content and language switching is reversible',()=>{
 language.value='en';
 assert.equal(t('陌生队名'),'陌生队名');
 assert.equal(t('我的剩余 01:23'),'Your time remaining: 01:23');
 assert.equal(t('确定关闭「队伍甲」？关闭后无法重新落座或开赛，房间记录仍会保留。'),'Close “队伍甲”? Seating and play will be disabled. The room record will be retained.');
 assert.equal(t('极限速通（3×3）'),'Extreme Speedrun (3×3)');
 assert.equal(t('黄16'),'Yellow 16');
 assert.equal(t('白5'),'White 5');
 assert.equal(t('项目 G 结果'),'Game G · Result');
 language.value='zh';assert.equal(t('我的剩余 01:23'),'我的剩余 01:23');
});
test('account language sync is read-only and local override wins until Auto is restored',async()=>{
 let calls=0;
 languageOverride.value=null;
 await syncAccountLanguage(async()=>{calls++;return {language:'en'};});
 assert.equal(language.value,'en');
 setLanguage('zh');
 await syncAccountLanguage(async()=>{calls++;return {language:'en'};});
 assert.equal(language.value,'zh');
 setLanguage('auto');
 assert.equal(language.value,'en');
 assert.ok(calls>=2);
 await syncAccountLanguage(async()=>{throw Error('offline');});
 assert.equal(language.value,'en');
 language.value='zh';
});
