import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { chickenPose, deformChickenPoint, createChickenMesh, CHICKEN_CYCLE_MS } from '../src/live/chickenMotion.js';
import { giftAsset } from '../src/live/giftArtwork.js';

test('reference photographs ship unaltered, including feather, expression and background details', () => {
  for (const [asset, original] of [['chicken-reference','幽默唤鸡'], ['chicken-balance-reference','幽默唤鸡2'], ['serious-reference','严肃唤鸡']]) {
    assert.deepEqual(readFileSync(new URL(`../public/live-gifts/${asset}.jpg`, import.meta.url)),
      readFileSync(new URL(`../../docs_and_configs/design/live_gifts/references/${original}.jpg`, import.meta.url)));
  }
  assert.equal(giftAsset('serious'), '/live-gifts/serious-reference.jpg');
});

test('continuous pose tracks loop at rest, not by swapping two photographs', () => {
  for (const id of ['chicken','serious']) {
    const duration = CHICKEN_CYCLE_MS[id];
    assert.deepEqual(chickenPose(id, 0), [0,0]);
    assert.deepEqual(chickenPose(id, duration), [0,0]);
    for (let time = 0; time < duration; time += 10) {
      const a = chickenPose(id, time), b = chickenPose(id, time + 1);
      assert.ok(a.every((value,i) => Math.abs(value-b[i]) < .015));
    }
    assert.notDeepEqual(chickenPose(id, duration*.4), [0,0]);
  }
});

test('background perimeter and supporting foot are pinned, with no folded mesh triangles', () => {
  const mesh = createChickenMesh();
  for (const id of ['chicken','serious']) {
    for (let time = 0; time < CHICKEN_CYCLE_MS[id]; time += 80) {
      const pose = chickenPose(id, time);
      for (const point of [[0,120],[240,120],[120,0],[120,240],[135,218]]) {
        assert.deepEqual(deformChickenPoint(id, ...point, pose), point);
      }
      const points = [];
      for (let i=0; i<mesh.points.length; i+=2) {
        const x=mesh.points[i], y=mesh.points[i+1];
        const p=deformChickenPoint(id,x,y,pose);
        assert.ok(Math.hypot(p[0]-x,p[1]-y) < 15);
        points.push(p);
      }
      for (let i=0; i<mesh.indices.length; i+=3) {
        const [a,b,c] = Array.from(mesh.indices.slice(i,i+3), index => points[index]);
        const area=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
        assert.ok(area < -12, `${id} mesh folded at ${time}`);
      }
    }
  }
});
