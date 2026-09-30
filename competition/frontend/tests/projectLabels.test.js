import test from 'node:test';
import assert from 'node:assert/strict';
import { ALL_PROJECTS, TOURNAMENT_PROJECTS, PROJECT_BY_ORDER } from '../src/projects/catalog.js';
import { competitionProjectLabel, competitionPhaseLabel } from '../../shared/projectLabels.mjs';

test('visible projects are numbered first without changing stable practice routes', () => {
  assert.deepEqual(TOURNAMENT_PROJECTS.map(project => project.displayOrder), Array.from({length:12}, (_, i) => i + 1));
  assert.deepEqual(ALL_PROJECTS.slice(12).map(project => project.displayOrder), [13,14,15,16,17,18,19,20]);
  assert.deepEqual(ALL_PROJECTS.slice(12).map(project => project.order), [2,4,8,10,12,15,18,20]);
  assert.equal(PROJECT_BY_ORDER[5].displayOrder, 3);
  assert.equal(PROJECT_BY_ORDER[5].practicePath, '/practice/5');
  assert.equal(PROJECT_BY_ORDER[19].displayOrder, 12);
});

test('every formal project has an English display name',()=>{
  for(const project of TOURNAMENT_PROJECTS){
    const label=competitionProjectLabel({project_ref:project.id,name:project.title},'en');
    assert.ok(label && !/[\u4e00-\u9fff]/.test(label),project.id);
  }
});
test('lobby stage labels follow selected language without raw status codes',()=>{
  for(const phase of ['DRAW','FIRST_PICK_BAN','SECOND_PICK_BAN','BLIND_PICK','C_DRAW','LINEUP','GAME_A_READY','GAME_B_PLAYING','GAME_C_RESULT','FINISHED','CANCELLED']){
    assert.ok(/[\u4e00-\u9fff]/.test(competitionPhaseLabel(phase,'zh')));
    assert.ok(!/[\u4e00-\u9fff_]/.test(competitionPhaseLabel(phase,'en')));
  }
});
