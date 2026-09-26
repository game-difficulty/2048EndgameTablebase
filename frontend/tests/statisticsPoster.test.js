import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import { statisticsChartExtent } from '../src/human/statisticsPoster.js';

test('statistics poster uses the selected game axis range', () => {
  const extent=statisticsChartExtent([
    {game_index:10,ended_at:100,pb_score:1000},
    {game_index:40,ended_at:900,pb_score:5000},
  ],['pb_score'],'games','score');
  assert.equal(extent.dataMinX,10);
  assert.equal(extent.dataMaxX,40);
});

test('statistics poster uses timestamps for the time setting', () => {
  const extent=statisticsChartExtent([
    {game_index:10,ended_at:100,pb_score:1000},
    {game_index:40,ended_at:900,pb_score:5000},
  ],['pb_score'],'time','score');
  assert.equal(extent.dataMinX,100);
  assert.equal(extent.dataMaxX,900);
});

test('statistics poster rate axis keeps at least twenty percentage points', () => {
  const extent=statisticsChartExtent([{game_index:10,value:.46},{game_index:11,value:.47}],['value'],'games','rate');
  assert.ok(extent.maxY-extent.minY>=.2-Number.EPSILON);
});

test('statistics poster ignores missing B10 values instead of plotting zero', () => {
  const extent=statisticsChartExtent([{game_index:1,pb_score:800,b10_score:null}],['pb_score','b10_score'],'games','score');
  assert.ok(extent.minY>0);
});

test('statistics poster rating extent includes negative early ratings', () => {
  const extent=statisticsChartExtent([{game_index:1,b10_rating:-120},{game_index:2,b10_rating:220}],['b10_rating'],'games','rating');
  assert.ok(extent.minY < -120);
  assert.ok(extent.maxY > 220);
});

test('statistics download waits for variant-specific data after switching', () => {
  const source=fs.readFileSync(new URL('../src/human/PlayerStatistics.vue',import.meta.url),'utf8');
  assert.match(source,/loading \|\| posterBusy \|\| !Object\.keys\(featureLabels\)\.length/);
  assert.match(source,/series\.value=\[\]; rateSeries\.value=\[\]; featureLabels\.value=\{\}/);
});
