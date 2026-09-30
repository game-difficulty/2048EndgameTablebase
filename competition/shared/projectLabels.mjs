import { englishProjectNames, projectMessages } from '../frontend/src/projectTranslations.js';

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
  ...englishProjectNames,
};
// Exact source-text translations keep archived/custom rules intact instead of
// replacing them with today's rules merely because the project ID matches.
const descriptions = {
  ...projectMessages,
  '50%概率生成4；双方死亡后按得分结算。': 'New tiles have a 50% chance of being a 4. Compare scores when both players have no legal moves remaining.',
  '只生成2，可重开；盘面和达到或超过1022即获胜。': 'Only 2s spawn. Restarts are allowed. Reach a tile sum of at least 1022 to win.',
  '256不可合并；双方死亡后按得分结算。': 'The 256 tiles cannot merge. Compare scores when both players have no legal moves remaining.',
  '每局使用随机12格棋盘；棋盘外区域不可进入，双方死亡后按得分结算。': 'Each run uses a randomly shaped 12-cell board. Tiles cannot leave it. Compare scores when both players have no legal moves remaining.',
  '64合成双格128；128相撞合成双格或三格256；双方死亡后按得分结算。': 'Two 64s form a two-cell 128. Colliding 128s form a two- or three-cell 256. Compare scores when both players have no legal moves remaining.',
  '炸弹移动后倒计时减少，归零变墙；双方死亡后按得分结算。': 'Bomb countdowns decrease after movement; at zero, bombs become walls. Compare scores when both players have no legal moves remaining.',
  '1024及以上数字块会分裂成两块并替代该步出数；双方死亡后比较盘面和。': 'Tiles of 1024 or higher split into two, replacing that move’s spawn. Compare tile sums when both players have no legal moves remaining.',
  '有效移动有机会改为撤销并出两个数；可重开，先合出2048者获胜。': 'A valid move may instead undo the previous move and spawn two tiles. Restarts are allowed. Be the first to make 2048.',
};
export function competitionProjectDescription(project, lang='zh') {
  const source=project?.description || '';
  return lang==='en' ? (descriptions[source] || source) : source;
}
const phases = {
  SEATING:['选手落座','Player seating'], READY_CHECK:['队长准备','Captain ready check'],
  DRAW:['先后手抽签','First-pick draw'], FIRST_PICK_BAN:['先手选禁','First pick / ban'],
  SECOND_PICK_BAN:['后手选禁','Second pick / ban'], BLIND_PICK:['双方盲选','Blind picks'],
  C_DRAW:['第三项目抽签','Game C draw'], LINEUP:['秘密布阵','Secret lineups'],
  FINISHED:['全场结算','Match finished'], CANCELLED:['比赛取消','Cancelled'],
};
export function competitionPhaseLabel(phase, lang='zh') {
  const game = /^GAME_([ABC])_(READY|PLAYING|RESULT)$/.exec(phase || '');
  if(game){const state={READY:['开局检查','Ready check'],PLAYING:['对局中','In progress'],RESULT:['单局结果','Game result']}[game[2]];return `${lang==='zh'?'项目':'Game'} ${game[1]} · ${state[lang==='zh'?0:1]}`;}
  return phases[phase]?.[lang==='zh'?0:1] || (lang==='zh'?'比赛准备中':'Preparing match');
}
export function competitionProjectLabel(project, lang='zh') {
  if(!project)return '';
  if(lang==='zh' && project.project_ref==='tournament-shape-shifter-hard-12')return '随机形状';
  return lang==='zh'?(project.name || project.project_name || ''):(names[project.project_ref] || project.name || project.project_name || 'Custom project');
}
