import test from 'node:test';
import assert from 'node:assert/strict';
import { TOURNAMENT_PROJECTS } from '../src/projects/catalog.js';
import { competitionProjectLabel, competitionPhaseLabel } from '../../shared/projectLabels.mjs';

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
