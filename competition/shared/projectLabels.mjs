const names = {
  'tournament-cargo-transport-4x4': 'Cargo Transport',
  'tournament-spawn4-50-3x3': '50% Fours (3×3)',
  'tournament-evil-spawn-4x4': 'Adversarial Spawn',
  'tournament-pure2-full-race-3x3': 'Pure-2 Full Board Race',
  'tournament-grand-full-undo-race-3x3': 'Grand Full Board Undo Race',
  'tournament-dice-wall-3x4': 'Dice Obstacles',
  'tournament-mirror-64x10-race-4x4': 'Mirror 64 × 10 Race',
  'tournament-256-brick-5x5': '256 Bricks',
  'tournament-isolated-island-hard-4x4': 'Hard Isolated Island',
  'tournament-shape-shifter-hard-12': 'Shape Shifter',
  'practice-hundred-step-seal-4x4': 'Hundred-Step Lockdown',
  'practice-growing-tiles-4x4': 'Growing Tiles',
  'practice-pair-bond-4x4': 'Pair Bond',
  'practice-chemical-reaction-4x4': 'Chemical Reaction',
  'practice-timed-bomb-4x4': 'Timed Bomb',
  'practice-full-load-4x4': 'Full Load',
  'practice-heavy-tiles-4x4': 'Heavy Tiles',
  'practice-fission-4x4': 'Fission',
  'practice-aftershock-4x4': 'Aftershock',
  'practice-look-back-3x4': 'Look Back',
  'standard-2048-test': 'Standard 2048',
};
const phases = {
  SEATING:['选手落座','Player seating'], READY_CHECK:['队长准备','Captain ready check'],
  DRAW:['先后手抽签','First-pick draw'], FIRST_PICK_BAN:['先手选禁','First pick / ban'],
  SECOND_PICK_BAN:['后手选禁','Second pick / ban'], BLIND_PICK:['双方盲选','Blind picks'],
  C_DRAW:['第三项目抽签','Game C draw'], LINEUP:['秘密布阵','Secret lineups'],
  FINISHED:['全场结算','Match finished'], CANCELLED:['比赛取消','Cancelled'],
};
export function competitionPhaseLabel(phase, lang='zh') {
  const game = /^GAME_([ABC])_(READY|PLAYING|RESULT)$/.exec(phase || '');
  if(game){const state={READY:['开局检查','Ready check'],PLAYING:['对局中','In progress'],RESULT:['结果确认','Result confirmation']}[game[2]];return `${lang==='zh'?'项目':'Game'} ${game[1]} · ${state[lang==='zh'?0:1]}`;}
  return phases[phase]?.[lang==='zh'?0:1] || (lang==='zh'?'比赛准备中':'Preparing match');
}
export function competitionProjectLabel(project, lang='zh') {
  if(!project)return '';
  if(lang==='zh' && project.project_ref==='tournament-shape-shifter-hard-12')return '随机形状';
  return lang==='zh'?(project.name || project.project_name || ''):(names[project.project_ref] || project.name || project.project_name || 'Custom project');
}
