const BASE_TOURNAMENT_PROJECTS = Object.freeze([
  {
    order: 1,
    practicePath: '/practice/1',
    id: 'tournament-cargo-transport-4x4',
    key: 'project-01',
    title: '真华容道（4×4）',
    shortTitle: '真华容道',
    description: '前10次有效移动先整理棋盘，随后从上方出现首个特殊块；特殊块整块滑到尽头，从下方中央送出。无路可走或10分钟到时结束，送出越多越好。',
    estimatedMinutes: '10 min',
    rows: 4, cols: 4, cargoTransport: true, timeLimitMs: 600000,
  },
  {
    order: 2,
    practicePath: '/practice/2',
    id: 'tournament-spawn4-50-3x3',
    key: 'project-02',
    title: '50%出4（3×3）',
    shortTitle: '50%出4',
    description: '每次出数有50%概率为4。一局定胜负，双方都无路可走后比较得分。',
    estimatedMinutes: '5–20 min',
    rows: 3, cols: 3, spawn4Rate: .5,
  },
  {
    order: 3,
    practicePath: '/practice/3',
    id: 'tournament-evil-spawn-4x4',
    key: 'project-03',
    title: '对抗出数（4×4）',
    shortTitle: '对抗出数',
    description: '每步由 EvilGen 选择更难的出数位置。一局定胜负，双方都无路可走后比较得分。',
    estimatedMinutes: '1–10 min',
    rows: 4, cols: 4, evilSpawn: true,
  },
  {
    order: 4,
    practicePath: '/practice/4',
    id: 'tournament-pure2-full-race-3x3',
    key: 'project-04',
    title: '纯2满盘竞速（3×3）',
    shortTitle: '纯2满盘竞速',
    description: '只生成2，允许重开。先将盘面和达到或超过1022即完成。',
    estimatedMinutes: '3–10 min',
    rows: 3, cols: 3, spawn4Rate: 0, targetSum: 1022, targetAtLeast: true, allowRestart: true, race: true,
  },
  {
    order: 5,
    practicePath: '/practice/5',
    id: 'tournament-grand-full-undo-race-3x3',
    key: 'project-05',
    title: '大满盘撤销竞速（3×3）',
    shortTitle: '大满盘撤销竞速',
    description: '标准出数，允许重开和撤销。先将盘面和精确做到2044即完成。',
    estimatedMinutes: '5–15 min',
    rows: 3, cols: 3, spawn4Rate: .1, targetSum: 2044, allowRestart: true, allowUndo: true, race: true,
  },
  {
    order: 6,
    practicePath: '/practice/6',
    id: 'tournament-dice-wall-3x3',
    key: 'project-06',
    title: '骰子障碍（3×3）',
    shortTitle: '骰子障碍',
    description: '开局掷骰：1–3在角位、4–5在边位、6在中心放墙。双方死亡后比较盘面和。',
    estimatedMinutes: '3–5 min',
    rows: 3, cols: 3, spawn4Rate: .1, diceWall: true, resultMetric: 'boardSum',
  },
  {
    order: 7,
    practicePath: '/practice/7',
    id: 'tournament-mirror-64x10-race-4x4',
    key: 'project-07',
    title: '镜面64砖×10竞速（4×4）',
    shortTitle: '镜面64砖×10',
    description: '中央十字是墙，四条外沿是传送门：左右相通、上下相通。64不可合并，先同时拥有10个64获胜。',
    estimatedMinutes: '2–15 min',
    rows: 4, cols: 4, spawn4Rate: .1, mirrorPortals: true, unmergeable: 64,
    targetTile: 64, targetCount: 10, allowRestart: true, race: true,
  },
  {
    order: 8,
    practicePath: '/practice/8',
    id: 'tournament-256-brick-5x5',
    key: 'project-08',
    title: '256砖（5×5）',
    shortTitle: '256砖',
    description: '5×5棋盘，256不可继续合并。一局定胜负，双方死亡后比较得分。',
    estimatedMinutes: '10–20 min',
    rows: 5, cols: 5, spawn4Rate: .1, unmergeable: 256,
  },
  {
    order: 9,
    practicePath: '/practice/9',
    id: 'tournament-isolated-island-hard-4x4',
    key: 'project-09',
    title: '困难孤岛（4×4）',
    shortTitle: '困难孤岛',
    description: '会生成只与同类合并的孤岛特殊块，合并孤岛不计分。双方都无路可走后比较得分。',
    estimatedMinutes: '待测',
    rows: 4, cols: 4, spawn4Rate: .1, isolatedIsland: true,
  },
  {
    order: 10,
    practicePath: '/practice/10',
    id: 'tournament-shape-shifter-hard-12',
    key: 'project-10',
    title: '随机形状棋盘（困难）',
    shortTitle: '随机形状',
    description: '每局使用随机生成的12格棋盘，棋盘外区域不可进入。双方都无路可走后比较得分。',
    estimatedMinutes: '待测',
    boardLabel: '随机 · 12格',
    rows: 6, cols: 6, spawn4Rate: .1, shapeShifter: true, playableCells: 12,
    adapterRulesVersion: 'tournament-v3',
  },
]);

const ADDITIONAL_TOURNAMENT_PROJECTS = Object.freeze([
  {
    order: 11,
    practicePath: '/practice/11',
    id: 'practice-hundred-step-seal-4x4',
    key: 'project-11',
    title: '百步封锁（4×4）',
    shortTitle: '百步封锁',
    description: '开局先随机封住3格，再生成初始数字；之后每100个有效移动轮换封锁。封住的数字保留，且不会在封锁格出数。无路可走后比较得分。',
    estimatedMinutes: '待测',
    rows: 4, cols: 4, spawn4Rate: .1,
    sealEveryMoves: 100, sealCount: 3,
  },
  {
    order: 12,
    practicePath: '/practice/12',
    id: 'practice-growing-tiles-4x4',
    key: 'project-12',
    title: '越来越大（4×5）',
    shortTitle: '越来越大',
    description: '64合成双格128；两个128移动时有格子重合，即合成双格或三格256。多格砖整块移动，256不可合并；死亡后比得分。',
    estimatedMinutes: '待测',
    rows: 4, cols: 5, spawn4Rate: .1, polyomino: true,
    adapterRulesVersion: 'tournament-v3',
  },
]);

export const TOURNAMENT_PROJECTS = Object.freeze([...BASE_TOURNAMENT_PROJECTS, ...ADDITIONAL_TOURNAMENT_PROJECTS]);

// These rules are being trialled on the practice site. They are intentionally
// absent from TOURNAMENT_PROJECTS until their competition adapters are ready.
export const PRACTICE_ONLY_PROJECTS = Object.freeze([
  { order: 13, practicePath: '/practice/13', id: 'practice-pair-bond-4x4', key: 'practice-13',
    title: '出双入对（4×4）', shortTitle: '出双入对',
    description: '偶尔出现特殊块；两块相邻便粘成双格，新的双格形成时旧双格消失。无路可走后比得分。',
    rows: 4, cols: 4, spawn4Rate: .1, specialSpawnRate: .05, specialRule: 'pair' },
  { order: 14, practicePath: '/practice/14', id: 'practice-chemical-reaction-4x4', key: 'practice-14',
    title: '化学反应（4×4）', shortTitle: '化学反应',
    description: '偶尔出现两色特殊块；同色相撞消失，异色相撞合成一格墙。无路可走后比得分。',
    rows: 4, cols: 4, spawn4Rate: .1, specialSpawnRate: .05, specialRule: 'chemical' },
  { order: 15, practicePath: '/practice/15', id: 'practice-timed-bomb-4x4', key: 'practice-15',
    title: '定时炸弹（4×4）', shortTitle: '定时炸弹',
    description: '偶尔出现倒计时为12–32的炸弹；每次移动减一，归零变墙。炸弹相撞合并倒计时。无路可走后比得分。',
    rows: 4, cols: 4, spawn4Rate: .1, specialSpawnRate: .05,
    bombCountdownMin: 12, bombCountdownMax: 32, specialRule: 'bomb' },
]);

export const PRACTICE_PROJECTS = Object.freeze([...TOURNAMENT_PROJECTS, ...PRACTICE_ONLY_PROJECTS]);

export const PROJECT_BY_ID = Object.freeze(Object.fromEntries(
  PRACTICE_PROJECTS.map(project => [project.id, Object.freeze(project)]),
));

export const PROJECT_BY_ORDER = Object.freeze(Object.fromEntries(
  PRACTICE_PROJECTS.map(project => [project.order, project]),
));

export const PROJECT_BY_TITLE = Object.freeze(Object.fromEntries(
  TOURNAMENT_PROJECTS.map(project => [project.title, project]),
));

export function competitionProjectInput(project) {
  const rulesVersion = project.adapterRulesVersion || 'tournament-v2';
  return {
    key: project.key,
    name: project.title,
    description: project.description,
    project_ref: project.id,
    adapter_rules_version: rulesVersion,
    rules_version: rulesVersion,
  };
}
