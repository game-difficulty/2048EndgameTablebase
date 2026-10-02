import { test } from 'node:test';
import assert from 'node:assert/strict';
import { aspectRatio, chooseRuleSize, fitBoard } from '../src/live/content/adaptiveBoardLayout.js';

test('intrinsic ratio accepts changing rectangular, square and cargo geometry', () => {
  for (const [value, expected] of [['4 / 7',4/7],['3 / 4',.75],['6 / 3',2],['auto 5 / 8',.625],['1',1],['auto',1]]) {
    assert.equal(aspectRatio({aspectRatio:value}),expected);
  }
});

test('aftershock bounding boxes fit both axes with square cells at tall and wide ratios', () => {
  for (const [rows,cols] of [[7,5],[4,8],[12,3],[3,12],[4,4]]) {
    const ratio=(cols+.1*(cols+1))/(rows+.1*(rows+1));
    const width=fitBoard(290,260,ratio);
    assert.ok(width<=290 && width/ratio<=260+1e-8);
    assert.ok(Math.abs(width/(cols+.1*(cols+1))-(width/ratio)/(rows+.1*(rows+1)))<1e-8);
  }
});
test('rules use the largest readable size and never truncate when budget is exhausted', () => {
  const measure = ({font,padding,lineHeight}) => font * lineHeight * 3 + padding * 2;
  assert.equal(chooseRuleSize(150,measure).font,24);
  assert.deepEqual(chooseRuleSize(120,measure),{font:24,padding:4,lineHeight:1.5,height:116});
  assert.equal(chooseRuleSize(85,measure).font,20);
  const fallback=chooseRuleSize(20,measure);
  assert.equal(fallback.font,18);
  assert.ok(fallback.height>20);
});
