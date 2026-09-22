const bi = (zh, en) => Object.freeze({ zh, en });

export const MINIGAME_TROPHY_THRESHOLDS = Object.freeze({
  tile: Object.freeze([1024, 2048, 4096, 8192]),
  pattern: Object.freeze([256, 512, 1024, 2048]),
  endless: Object.freeze([50000, 120000, 300000, 600000]),
  hybrid: Object.freeze([30000, 60000, 120000, 240000]),
});

const COPY = Object.freeze({
  zh: Object.freeze({
    title: '规则与奖杯',
    details: '查看完整规则',
    objective: '游戏目标',
    specialRules: '特殊规则',
    difficulty: '当前难度',
    casual: '休闲',
    hard: '困难',
    trophies: '奖杯条件',
    metric: '评定指标',
    tiers: ['铜', '银', '金', '大奖杯'],
    retained: '已获得的奖杯与代币会保留；新规则只会让奖杯升级，不会降级。',
    tileMetric: '整局历史最高方块',
    patternMetric: '完成目标图案时的基准值 S',
    scoreMetric: '整局累计分数',
    powerups: (hard, blitz) => blitz
      ? `本局三种道具各有 10 个。单次有效移动增加 300–1023 分时，有 ${hard ? 5 : 25}% 概率再获得一个随机道具。`
      : `本局三种道具初始各有 ${hard ? 1 : 5} 个。单次有效移动增加 300–1023 分时，有 ${hard ? 5 : 25}% 概率再获得一个随机道具。`,
    coreSame: '本游戏的核心特殊规则在两种难度下相同。',
  }),
  en: Object.freeze({
    title: 'Rules & Trophies',
    details: 'View full rules',
    objective: 'Objective',
    specialRules: 'Special rules',
    difficulty: 'Current difficulty',
    casual: 'Casual',
    hard: 'Hard',
    trophies: 'Trophy requirements',
    metric: 'Rating metric',
    tiers: ['Bronze', 'Silver', 'Gold', 'Grand'],
    retained: 'Trophies and tokens already earned are retained. The new rules can only upgrade a trophy, never downgrade it.',
    tileMetric: 'Highest tile reached during the run',
    patternMetric: 'Base value S of a completed target pattern',
    scoreMetric: 'Cumulative score during the run',
    powerups: (hard, blitz) => blitz
      ? `The run starts with 10 of each power-up. A valid move worth 300–1023 points has a ${hard ? 5 : 25}% chance to award one random power-up.`
      : `The run starts with ${hard ? 1 : 5} of each power-up. A valid move worth 300–1023 points has a ${hard ? 5 : 25}% chance to award one random power-up.`,
    coreSame: 'The core variant rules are the same in both difficulties.',
  }),
});

const GAME_RULES = Object.freeze({
  'design-master-1': {
    trophyKind: 'pattern',
    summary: bi('在中央 2×2 区域完成指定比例图案。', 'Complete the required ratio pattern in the central 2×2 area.'),
    objective: bi('完成当前标注的目标图案；每次完成后，整个目标翻倍。', 'Complete the currently marked target pattern. After each completion, every target value doubles.'),
    mechanics: [
      bi('设基准值为 S，中央 2×2 的目标为「2S、S / S、2S」；初始 S=256。', 'With base value S, the central 2×2 target is 2S, S / S, 2S. The first target uses S=256.'),
      bi('只检查有目标标记的格子；其他格子可以放置任意数字。', 'Only marked cells are checked; unmarked cells may contain any values.'),
    ],
  },
  'mystery-merge-1': {
    trophyKind: 'tile',
    summary: bi('大部分数字被隐藏，需要根据新方块和合并结果推断盘面。', 'Most values are hidden; infer the board from new tiles and merge results.'),
    objective: bi('在信息不完整的盘面上尽可能合成更高方块。', 'Build the highest tile possible while most board values are concealed.'),
    mechanics: [
      bi('新生成的方块可见；上一次移动中产生合并的位置会显示，其他非空格显示为未知。', 'The newest tile is visible, as are positions that merged on the previous move; other occupied cells are shown as unknown.'),
      bi('按住「查看」可暂时显示全部数字。', 'Hold Peek to reveal all values temporarily.'),
    ],
    difficulty: {
      casual: bi('查看次数无限。', 'Peek can be used without limit.'),
      hard: bi('开局可查看 1 次，每累计 10000 分再增加 1 次。', 'Start with 1 Peek; gain 1 additional use for every 10,000 cumulative points.'),
    },
  },
  'column-chaos': {
    trophyKind: 'tile',
    summary: bi('随机两列会定期整列交换。', 'Two random columns periodically swap in full.'),
    objective: bi('在列位置不断变化的盘面上合成更高方块。', 'Build high tiles while the column layout repeatedly changes.'),
    mechanics: [bi('只有有效移动才计入倒计。倒计归零时，随机选择两列并交换其全部内容。', 'Only valid moves reduce the counter. At zero, two random columns exchange all of their cells.')],
    difficulty: {
      casual: bi('每 40 次有效移动交换一次。', 'A swap occurs every 40 valid moves.'),
      hard: bi('每 30 次有效移动交换一次。', 'A swap occurs every 30 valid moves.'),
    },
  },
  'gravity-twist-1': {
    trophyKind: 'tile',
    summary: bi('每次移动后，所有方块都会再向下坠落。', 'After every move, all tiles fall downward again.'),
    objective: bi('利用固定向下的重力保持盘面可合并。', 'Use the permanent downward gravity to keep the board mergeable.'),
    mechanics: [bi('完成一次有效移动并生成新方块后，所有方块向下压紧；这次坠落不会产生合并。', 'After a valid move and spawn, all tiles compress downward. This gravity step never merges tiles.')],
  },
  blitzkrieg: {
    trophyKind: 'tile',
    blitz: true,
    summary: bi('在 3 分钟倒计内快速得分，合成 1024 可延长时间。', 'Score quickly within a three-minute timer; creating 1024 tiles extends the clock.'),
    objective: bi('在时间归零或无法移动之前取得尽可能高的分数和方块。', 'Earn the highest score and tile before time expires or no moves remain.'),
    mechanics: [
      bi('倒计从第一次操作开始，初始为 3 分钟。', 'The timer starts with the first move and begins at three minutes.'),
      bi('当盘面上 1024 方块的数量因合并而增加时，每个新 1024 奖励一次时间。', 'Whenever a merge increases the number of 1024 tiles on the board, each new 1024 grants a time bonus.'),
      bi('奖杯在本局结束时按历史最高方块结算，不是按总分结算。', 'Trophies are settled at the end of the run from the highest tile reached, not from total score.'),
    ],
    difficulty: {
      casual: bi('每个新 1024 增加 60 秒。', 'Each new 1024 adds 60 seconds.'),
      hard: bi('每个新 1024 增加 45 秒。', 'Each new 1024 adds 45 seconds.'),
    },
  },
  'tricky-tiles': {
    trophyKind: 'tile',
    summary: bi('部分新方块会由对抗 AI 选择尽可能不利的生成位置。', 'Some new tiles are placed by an adversarial AI in an inconvenient position.'),
    objective: bi('对抗恶意生成位置并尽可能合成更高方块。', 'Survive hostile spawns and build the highest tile possible.'),
    mechanics: [bi('每局开始时会在对应范围内随机确定一个对抗生成概率；其余新方块正常随机生成。', 'At the start of each run, one adversarial-spawn probability is chosen from the difficulty range; all other spawns are normal random spawns.')],
    difficulty: {
      casual: bi('对抗 AI 生成概率约为 33%–41.3%。', 'Adversarial-spawn probability is approximately 33–41.3%.'),
      hard: bi('对抗 AI 生成概率约为 43%–51.3%。', 'Adversarial-spawn probability is approximately 43–51.3%.'),
    },
  },
  'design-master-2': {
    trophyKind: 'pattern',
    summary: bi('让主对角线的四个格子同时达到目标值。', 'Make all four cells on the main diagonal equal the target value.'),
    objective: bi('完成当前对角线目标；每次完成后，目标整体翻倍。', 'Complete the current diagonal target; after each completion, the target doubles.'),
    mechanics: [
      bi('设基准值为 S，从左上到右下的四个格子都必须为 S；初始 S=256。', 'With base value S, all four cells from top-left to bottom-right must equal S. The first target uses S=256.'),
      bi('只检查标注的对角线格子。', 'Only the marked diagonal cells are checked.'),
    ],
  },
  'shape-shifter': {
    trophyKind: 'tile',
    summary: bi('每局使用一个随机生成的 18 格连通不规则棋盘。', 'Every run uses a randomly generated connected irregular board with 18 playable cells.'),
    objective: bi('适应当局棋盘形状，在受限的通道中合成更高方块。', 'Adapt to the generated shape and merge through its restricted lanes.'),
    mechanics: [bi('棋盘先在 12×12 网格中生成 18 个相连可玩格，再裁切到最小外接矩形；阻塞格不能进入。', 'Eighteen connected cells are generated inside a 12×12 grid and cropped to their minimum bounding rectangle; blocked cells cannot be entered.')],
    difficulty: {
      casual: bi('生成形状的最大完整矩形面积为 8–13 格。', 'The largest full rectangle in the shape has an area of 8–13 cells.'),
      hard: bi('生成形状的最大完整矩形面积为 6–11 格。', 'The largest full rectangle in the shape has an area of 6–11 cells.'),
    },
  },
  'ferris-wheel': {
    trophyKind: 'tile',
    summary: bi('外圈 12 格会定期顺时针轮转一格。', 'The twelve cells on the outer ring periodically rotate one step clockwise.'),
    objective: bi('在外圈周期性轮转的情况下维持数字链并合成更高方块。', 'Maintain merge chains while the outer ring periodically rotates.'),
    mechanics: [bi('只有有效移动才计入倒计。轮转只改变外圈，中央 2×2 保持不变。', 'Only valid moves reduce the counter. Rotation affects only the outer ring; the central 2×2 area stays in place.')],
    difficulty: {
      casual: bi('每 40 次有效移动轮转一次。', 'The ring rotates every 40 valid moves.'),
      hard: bi('每 30 次有效移动轮转一次。', 'The ring rotates every 30 valid moves.'),
    },
  },
  'gravity-twist-2': {
    trophyKind: 'tile',
    summary: bi('每次移动后，盘面会再自动向下移动并合并。', 'After every move, the board automatically moves downward and may merge.'),
    objective: bi('利用会合并的二次重力移动构建高值方块。', 'Use the second, merging gravity move to build high-value tiles.'),
    mechanics: [bi('完成一次有效移动并生成新方块后，系统再执行一次标准向下移动；该步可合并且会计分。', 'After a valid move and spawn, the game performs one standard downward move. It may merge tiles and its merge points are added to the score.')],
  },
  'design-master-3': {
    trophyKind: 'pattern',
    summary: bi('让三个分散锚点同时达到目标值。', 'Make three separated anchor cells equal the target value.'),
    objective: bi('完成当前三锚点目标；每次完成后，目标整体翻倍。', 'Complete the current three-anchor target; after each completion, the target doubles.'),
    mechanics: [
      bi('设基准值为 S，第 1 行第 3 格、第 3 行第 1 格、第 4 行第 4 格都必须为 S；初始 S=256。', 'With base value S, row 1 column 3, row 3 column 1, and row 4 column 4 must all equal S. The first target uses S=256.'),
      bi('只检查三个标注锚点。', 'Only the three marked anchors are checked.'),
    ],
  },
  'mystery-merge-2': {
    trophyKind: 'tile',
    summary: bi('新生成的方块被隐藏，只有参与合并后才会显露。', 'New tiles are hidden and are revealed only after taking part in a merge.'),
    objective: bi('记忆新方块的位置与可能数值，尽可能合成更高方块。', 'Track hidden spawns and build the highest tile possible.'),
    mechanics: [
      bi('每个新生成的数字都会被遮住；它参与一次合并后，合并结果会变为可见。', 'Every new spawn is concealed. Once it participates in a merge, the resulting tile becomes visible.'),
      bi('按住「查看」可暂时显示全部数字。', 'Hold Peek to reveal all values temporarily.'),
    ],
    difficulty: {
      casual: bi('查看次数无限。', 'Peek can be used without limit.'),
      hard: bi('开局可查看 1 次，每累计 10000 分再增加 1 次。', 'Start with 1 Peek; gain 1 additional use for every 10,000 cumulative points.'),
    },
  },
  'ice-age': {
    trophyKind: 'tile',
    summary: bi('长时间停在原位的方块会冻结为不可移动的障碍。', 'Tiles that remain in place for too long freeze into immovable obstacles.'),
    objective: bi('持续让重要方块发生移动或合并，避免盘面被冰块切断。', 'Keep important tiles moving or merging so frozen obstacles do not divide the board.'),
    mechanics: [
      bi('方块在有效移动中保持原位时，其冻结计数增加；一旦发生移动，计数清零。', 'A tile gains one freeze step whenever it stays in place during a valid move; moving it resets its counter.'),
      bi('达到冻结步数后，该方块保留显示数值，但不再移动或合并。', 'At the freeze limit, the tile keeps its displayed value but can no longer move or merge.'),
    ],
    difficulty: {
      casual: bi('同一方块连续静止 100 步后冻结。', 'A tile freezes after remaining still for 100 consecutive steps.'),
      hard: bi('同一方块连续静止 80 步后冻结。', 'A tile freezes after remaining still for 80 consecutive steps.'),
    },
  },
  'isolated-island': {
    trophyKind: 'tile',
    summary: bi('盘面会生成只能与同类相遇的孤岛格。', 'The board can spawn island cells that interact only with another island.'),
    objective: bi('绕开孤岛对路线的干扰，并尽可能合成更高方块。', 'Route around islands and build the highest tile possible.'),
    mechanics: [
      bi('孤岛不能与数字合并；两个孤岛相遇时可合为一个孤岛。', 'An island cannot merge with numbered tiles. Two islands can combine into one island.'),
      bi('生成概率会随当前孤岛数量增加而下降，每个现存孤岛降低 2 个百分点。', 'Spawn chance falls by two percentage points for every island currently on the board.'),
    ],
    difficulty: {
      casual: bi('无孤岛时的基础生成率为 4%。', 'Base island-spawn chance is 4% when no island is present.'),
      hard: bi('无孤岛时的基础生成率为 5%。', 'Base island-spawn chance is 5% when no island is present.'),
    },
  },
  'design-master-4': {
    trophyKind: 'pattern',
    summary: bi('在右下 3×3 区域完成一个非对称的多倍数图案。', 'Complete an asymmetric multi-value pattern in the lower-right 3×3 area.'),
    objective: bi('完成当前多倍数目标；每次完成后，所有目标值翻倍。', 'Complete the current multi-value target; after each completion, every target value doubles.'),
    mechanics: [
      bi('设基准值为 S，目标 3×3 为「S/128、S、S/2 / S/64、2S、S/4 / S/32、S/16、S/8」；初始 S=256。', 'With base value S, the target 3×3 is S/128, S, S/2 / S/64, 2S, S/4 / S/32, S/16, S/8. The first target uses S=256.'),
      bi('只检查标注的九个格子。', 'Only the nine marked cells are checked.'),
    ],
  },
  'endless-factorization': {
    trophyKind: 'endless',
    summary: bi('使用因子分解特殊对象降低方块，持续腾出空间得分。', 'Use factorization objects to reduce tiles, reopen space, and keep scoring.'),
    objective: bi('利用因子分解延长对局，累积尽可能高的总分。', 'Use factorization to extend the run and accumulate the highest possible score.'),
    mechanics: [
      bi('特殊对象会随操作方向滑动，碰到该行或列的首个数字时触发。', 'The special object slides with the chosen direction and triggers on the first numbered tile it encounters in that row or column.'),
      bi('若命中方块的指数为 e>1，它会被拆成两个指数之和为 e 的方块；命中 2 时直接消失。', 'A hit tile with exponent e>1 is split into two tiles whose exponents add to e; a hit value of 2 disappears.'),
      bi('对象消耗后，每次新方块生成有 3% 概率再生成一个。', 'After the object is consumed, each spawn has a 3% chance to create another one.'),
    ],
  },
  'endless-explosions': {
    trophyKind: 'endless',
    summary: bi('操作可移动炸弹消除数字，从而持续腾出空间。', 'Move bombs into numbered tiles to clear space and continue the run.'),
    objective: bi('利用炸弹延长对局，累积尽可能高的总分。', 'Use bombs to extend the run and accumulate the highest possible score.'),
    mechanics: [
      bi('炸弹会随操作方向滑动，并消除该行或列上碰到的首个数字。', 'A bomb slides with the chosen direction and destroys the first numbered tile it encounters in that row or column.'),
      bi('炸弹消耗后，每次新方块生成有 3% 概率再生成一个。', 'After a bomb is consumed, each spawn has a 3% chance to create another bomb.'),
    ],
  },
  'endless-giftbox': {
    trophyKind: 'endless',
    summary: bi('礼盒会把碰到的数字变成另一个随机数字。', 'Giftboxes transform the numbered tile they hit into another random value.'),
    objective: bi('利用礼盒变换延长对局，累积尽可能高的总分。', 'Use giftbox transformations to extend the run and accumulate the highest possible score.'),
    mechanics: [
      bi('礼盒会随操作方向滑动，命中后将该方块变为 4–1024 之间的随机 2 的幂，且不会保持原值。', 'A giftbox slides with the chosen direction and changes the hit tile to a random power of two from 4 through 1024, never keeping its original value.'),
      bi('较小数字的抽取概率更高。礼盒消耗后，每次新方块生成有 3% 概率再生成一个。', 'Smaller values are more likely. After the giftbox is consumed, each spawn has a 3% chance to create another one.'),
    ],
  },
  'endless-hybrid': {
    trophyKind: 'hybrid',
    summary: bi('混合炸弹、因子分解、礼盒和孤岛的无尽模式。', 'An endless mix of bombs, factorization, giftboxes, and islands.'),
    objective: bi('适应随机特殊对象，延长对局并累积尽可能高的总分。', 'Adapt to random special objects, extend the run, and accumulate the highest possible score.'),
    mechanics: [
      bi('每个特殊对象会随机选为炸弹、因子分解或礼盒，并按各自规则处理命中的数字。', 'Each special object is randomly a bomb, factorizer, or giftbox and resolves the hit tile with that object\'s rules.'),
      bi('特殊对象消耗后，每次新方块生成有 5% 概率再生成一个。', 'After the object is consumed, each spawn has a 5% chance to create another special object.'),
      bi('此模式还会生成只能与同类合并的孤岛，但不包含空袭。', 'This mode also spawns islands that merge only with islands; it does not include airstrikes.'),
    ],
    difficulty: {
      casual: bi('无孤岛时，孤岛基础生成率为 3%；每个现存孤岛降低 2 个百分点。', 'With no island present, base island-spawn chance is 3%; each existing island reduces it by two percentage points.'),
      hard: bi('无孤岛时，孤岛基础生成率为 4.5%；每个现存孤岛降低 2 个百分点。', 'With no island present, base island-spawn chance is 4.5%; each existing island reduces it by two percentage points.'),
    },
  },
  'endless-airraid': {
    trophyKind: 'endless',
    summary: bi('空袭标记会将下一步进入目标格的方块炸成临时弹坑。', 'An airstrike marker turns a tile entering its target on the next move into a temporary crater.'),
    objective: bi('利用空袭清理方块同时避免弹坑堵死盘面，累积尽可能高的总分。', 'Use airstrikes to clear tiles without letting craters choke the board, and accumulate the highest possible score.'),
    mechanics: [
      bi('有足够空格时，新方块生成后可能在一个空格出现目标标记。标记只持续到下一次有效移动。', 'When enough empty cells remain, a target may appear on an empty cell after a spawn. It lasts until the next valid move only.'),
      bi('若下一步后目标格被数字占据，该数字被消除并留下不可移动的弹坑；倒计结束后弹坑恢复为空格。', 'If the target is occupied after that move, the tile is destroyed and replaced by an immovable crater. The crater becomes empty when its counter expires.'),
      bi('目标基础生成率为 8%，现存弹坑会降低概率，但不低于 1%。', 'Base target chance is 8%; existing craters reduce it, with a minimum chance of 1%.'),
    ],
    difficulty: {
      casual: bi('弹坑持续 60 次有效移动。', 'A crater lasts for 60 valid moves.'),
      hard: bi('弹坑持续 100 次有效移动。', 'A crater lasts for 100 valid moves.'),
    },
  },
});

export const MINIGAME_RULE_IDS = Object.freeze(Object.keys(GAME_RULES));

function languageFor(locale) {
  return String(locale || '').toLowerCase().startsWith('zh') ? 'zh' : 'en';
}

function localized(value, language) {
  return value?.[language] || value?.en || '';
}

function trophyRows(kind, language) {
  const copy = COPY[language];
  const values = MINIGAME_TROPHY_THRESHOLDS[kind] || MINIGAME_TROPHY_THRESHOLDS.tile;
  const locale = language === 'zh' ? 'zh-CN' : 'en-US';
  return values.map((value, index) => {
    const number = Number(value).toLocaleString(locale);
    const prefix = kind === 'pattern'
      ? (index === values.length - 1 ? 'S ≥ ' : 'S = ')
      : '≥ ';
    return { key: index + 1, label: copy.tiers[index], requirement: `${prefix}${number}` };
  });
}

export function getMinigameRuleView(gameId, difficulty, locale = 'en') {
  const language = languageFor(locale);
  const copy = COPY[language];
  const spec = GAME_RULES[String(gameId || '')] || {
    trophyKind: 'tile',
    summary: bi('按照特殊规则进行 2048 对局。', 'Play 2048 with this variant\'s special rules.'),
    objective: bi('尽可能合成更高方块。', 'Build the highest tile possible.'),
    mechanics: [],
  };
  const hard = Number(difficulty) === 1;
  const difficultyCopy = spec.difficulty
    ? localized(hard ? spec.difficulty.hard : spec.difficulty.casual, language)
    : copy.coreSame;
  const metric = spec.trophyKind === 'pattern'
    ? copy.patternMetric
    : (spec.trophyKind === 'endless' || spec.trophyKind === 'hybrid' ? copy.scoreMetric : copy.tileMetric);
  return {
    title: copy.title,
    detailsLabel: copy.details,
    summary: localized(spec.summary, language),
    objectiveTitle: copy.objective,
    objective: localized(spec.objective, language),
    specialRulesTitle: copy.specialRules,
    mechanics: (spec.mechanics || []).map((item) => localized(item, language)),
    difficultyTitle: copy.difficulty,
    difficultyName: hard ? copy.hard : copy.casual,
    difficultyRules: [difficultyCopy, copy.powerups(hard, Boolean(spec.blitz))],
    trophiesTitle: copy.trophies,
    trophyMetricLabel: copy.metric,
    trophyMetric: metric,
    trophies: trophyRows(spec.trophyKind, language),
    retained: copy.retained,
  };
}

export function formatMinigameRules(gameId, difficulty, locale = 'en') {
  const view = getMinigameRuleView(gameId, difficulty, locale);
  const bulletLines = (items) => items.map((item) => `• ${item}`).join('\n');
  return [
    `${view.objectiveTitle}\n${view.objective}`,
    `${view.specialRulesTitle}\n${bulletLines(view.mechanics)}`,
    `${view.difficultyTitle} · ${view.difficultyName}\n${bulletLines(view.difficultyRules)}`,
    `${view.trophiesTitle} · ${view.trophyMetricLabel}: ${view.trophyMetric}\n${view.trophies.map((row) => `${row.label}: ${row.requirement}`).join('\n')}`,
    view.retained,
  ].filter(Boolean).join('\n\n');
}
