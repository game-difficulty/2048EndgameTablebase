import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
const source = readFileSync(new URL('../public/verse-replay/i18n.js', import.meta.url),'utf8');
function locale(language, search='') {
  const window = {location:{search}, navigator:{language}};
  vm.runInNewContext(source,{window,URLSearchParams});
  return window.ReplayI18n;
}
test('English uses playback and 2048 terminology', () => {
  const {translate:t} = locale('en-US');
  for(const [zh,en] of [['分数','Score'],['进度','Progress'],['节点用时','Tile milestones'],['原始步速 · 1×','Recorded timing · 1×'],['恒定步速 · 100 ms','Fixed interval · 100 ms/move']]) assert.equal(t(zh),en);
  assert.equal(t('AI 直播对局 · 4×4 · 1,234 步'),'AI livestream game · 4×4 · 1,234 moves');
  assert.equal(t('第 12 步 left 没有改变棋盘。'),'Move 12 does not change the board.');
  assert.equal(t('无法载入直播回放：回放已过期或不存在'),'Could not load livestream replay: This replay has expired or could not be found.');
});
test('Chinese is preserved and URL language overrides browser preference', () => {
  assert.equal(locale('zh-CN').translate('原始步速 · 1×'),'原始步速 · 1×');
  assert.equal(locale('zh-CN','?lang=en').translate('分数'),'Score');
  assert.equal(locale('en-US','?lang=zh').translate('分数'),'分数');
});
