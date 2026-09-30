import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ALL_PROJECTS } from '../../competition/frontend/src/projects/catalog.js';
import { competitionProjectLabel, competitionProjectDescription, competitionPhaseLabel } from '../../competition/shared/projectLabels.mjs';

test('all competition project names and rules switch languages without modifying source data', () => {
  for (const item of ALL_PROJECTS) {
    const project={project_ref:item.id,name:item.title,description:item.description};
    const before=JSON.stringify(project);
    assert.doesNotMatch(competitionProjectLabel(project,'en'),/[一-龥]/,item.id);
    assert.doesNotMatch(competitionProjectDescription(project,'en'),/[一-龥]/,item.id);
    assert.equal(competitionProjectDescription(project,'zh'),item.description);
    assert.equal(JSON.stringify(project),before);
  }
  assert.equal(competitionProjectLabel({project_ref:'tournament-grand-full-undo-race-3x3'},'en'),'Extreme Speedrun');
});
test('custom and archived rule text is preserved, not silently replaced by current rules',()=>{
  const old={project_ref:'tournament-cargo-transport-4x4',description:'An archived ten-minute match rule'};
  assert.equal(competitionProjectDescription(old,'en'),old.description);
  assert.equal(competitionProjectLabel({name:'Custom Cup Game'},'en'),'Custom Cup Game');
  assert.match(competitionProjectDescription({description:'炸弹移动后倒计时减少，归零变墙；双方死亡后按得分结算。'},'en'),/^Bomb countdowns/);
});
test('all broadcast phases have Chinese and English labels',()=>{
  for(const phase of ['SEATING','READY_CHECK','DRAW','FIRST_PICK_BAN','SECOND_PICK_BAN','BLIND_PICK','C_DRAW','LINEUP','FINISHED','CANCELLED',...['A','B','C'].flatMap(g=>['READY','PLAYING','RESULT'].map(s=>`GAME_${g}_${s}`))]){
    assert.doesNotMatch(competitionPhaseLabel(phase,'en'),/[一-龥]/);
    assert.match(competitionPhaseLabel(phase,'zh'),/[一-龥]/);
  }
});
test('shared avatar does not require tournament-only translation injection',()=>{
  const source=readFileSync(new URL('../../competition/frontend/src/PlayerAvatar.vue',import.meta.url),'utf8');
  assert.doesNotMatch(source,/\$t\(/);
});
