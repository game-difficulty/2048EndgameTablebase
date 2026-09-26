// The profile preview and downloaded PNG share this renderer.
// Coordinates follow the 1214 × 2048 reference, exported at 1600 × 2700.
export const POSTER_WIDTH = 1214;
export const POSTER_HEIGHT = 2048;
export const LANDSCAPE_POSTER_WIDTH = 2048;
export const LANDSCAPE_POSTER_HEIGHT = 1214;
const FONT = '"Clear Sans", "Microsoft YaHei", Arial, sans-serif';
const VARIANTS = { '4x4': [4, 4], '3x4': [3, 4], '2x4': [2, 4], '3x3': [3, 3] };
const COPY = {
  zh: { player: '玩家', pbRank: 'PB排名', raRank: 'RA排名', finalScore: '终局分数',
    boardSum: '盘面和', singleRating: '单局 Rating', fourRate: '出4率', empty: '暂无记录',
    quote1: '每一次合并', quote2: '都是向更高的自己靠近。',
    footer: '生成于 https://play.2048tables.online/' },
  en: { player: 'Player', pbRank: 'PB RANK', raRank: 'RA RANK', finalScore: 'FINAL SCORE',
    boardSum: 'BOARD SUM', singleRating: 'GAME RATING', fourRate: '4 SPAWN RATE', empty: 'NO RECORD',
    quote1: 'EVERY MERGE', quote2: 'BRINGS YOU CLOSER TO A HIGHER BEST.',
    footer: 'Generated at https://play.2048tables.online/' },
};

const palettes = {
  light: {
    page: '#fbf7f0', header: '#75685f', headerText: '#ffffff', stats: '#fff6e8',
    statsText: '#30241f', statsLine: '#cabfb3', orange: '#f2b179', mint: '#a9c8c2',
    card: '#fffbf5', cardLine: '#e2d9ce', cardShadow: '#ded5ca', text: '#30221d', muted: '#655e58',
    gold: '#b4871b', goldEdge: '#e9c45c', goldShadow: '#c8aa61', crown: '#fff8df',
    silver: '#92989d', bronze: '#a87550', rank: '#aaa198', board: '#bbada0', empty: '#cdc1b4', quote: '#f8f1e9',
  },
  dark: {
    page: '#1d1c20', header: '#474249', headerText: '#fff9f0', stats: '#343238',
    statsText: '#fff5e9', statsLine: '#696269', orange: '#e8a368', mint: '#9ac3bd',
    card: '#2b292e', cardLine: '#514b4d', cardShadow: '#121115', text: '#fff4e7', muted: '#c9beb3',
    gold: '#b58c31', goldEdge: '#d6b75e', goldShadow: '#6d5728', crown: '#fff4cf',
    silver: '#7f858c', bronze: '#986b4f', rank: '#736c70', board: '#80756f', empty: '#625a55', quote: '#332d2e',
  },
};

const defaultTilePalette = {
  2: ['#f9f1e8', '#756e67'], 4: ['#f7ead5', '#756e67'],
  8: ['#f2b179', '#ffffff'], 16: ['#f2b179', '#ffffff'],
  32: ['#f67c5f', '#ffffff'], 64: ['#f65e3b', '#ffffff'],
  128: ['#f1d271', '#ffffff'], 256: ['#f1d271', '#ffffff'],
  512: ['#f3ca4e', '#ffffff'], 1024: ['#f3c43c', '#ffffff'],
  2048: ['#f6ca2d', '#ffffff'], 4096: ['#6500a4', '#ffffff'],
  8192: ['#8700c3', '#ffffff'], 16384: ['#4c0073', '#ffffff'],
};
const formatNumber = value => value != null && Number.isFinite(Number(value))
  ? new Intl.NumberFormat('en-US').format(Number(value)) : '—';
const formatRating = value => value != null && Number.isFinite(Number(value))
  ? formatNumber(Math.round(Number(value))) : '—';
const dateText = seconds => {
  if (!seconds) return '—';
  const date = new Date(Number(seconds) * 1000);
  if (Number.isNaN(date.getTime())) return '—';
  const part = n => String(n).padStart(2, '0');
  return `${date.getFullYear()}-${part(date.getMonth() + 1)}-${part(date.getDate())} ${part(date.getHours())}:${part(date.getMinutes())}`;
};
const boardSum = board => Array.isArray(board) ? board.reduce((sum, value) => sum + (Number(value) || 0), 0) : null;
export function calculateFourSpawnRate(board, score) {
  if (!Array.isArray(board) || !Number.isFinite(Number(score))) return null;
  let sum = 0, weighted = 0;
  for (const raw of board) {
    const value = Number(raw) || 0;
    if (value < 0 || (value && !Number.isInteger(Math.log2(value)))) return null;
    sum += value;
    if (value) weighted += value * (Math.log2(value) - 1);
  }
  const fourSpawns = (weighted - Number(score)) / 4;
  const twoSpawns = (sum - (weighted - Number(score))) / 2;
  const totalSpawns = fourSpawns + twoSpawns;
  if (fourSpawns < 0 || twoSpawns < 0 || totalSpawns <= 0) return null;
  const rate = fourSpawns / totalSpawns;
  return Number.isFinite(rate) && rate >= 0 && rate <= 1 ? rate : null;
}
const fourRate = entry => {
  const serverRate = Number(entry?.four_spawn_rate);
  const derived = entry?.four_spawn_rate != null && Number.isFinite(serverRate) && serverRate >= 0 && serverRate <= 1
    ? serverRate : calculateFourSpawnRate(entry?.board, entry?.score);
  if (derived != null) return `${(100 * derived).toFixed(1)}%`;
  return '—';
};

export function posterCardBounds(index, layout = 'portrait') {
  if (layout === 'landscape') {
    if (index === 0) return { x: 32, y: 226, width: 780, height: 956 };
    if (index < 1 || index > 9) return null;
    if (index <= 2) return { x: 830 + (index - 1) * 602, y: 226, width: 584, height: 330 };
    const position = index - 3;
    return { x: 830 + (position % 4) * 300, y: position < 4 ? 574 : 887,
      width: 286, height: 295 };
  }
  if (index === 0) return { x: 47, y: 245, width: 1121, height: 521 };
  if (index < 1 || index > 9) return null;
  const position = index - 1;
  return { x: position % 2 ? 618 : 69, y: 780 + Math.floor(position / 2) * 246,
    width: 530, height: 237 };
}
function rounded(ctx, x, y, w, h, radius, color) {
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.roundRect(x, y, w, h, radius); ctx.fill();
}
function line(ctx, x1, y1, x2, y2, color, width = 1) {
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
}
function label(ctx, value, x, y, size, color, options = {}) {
  ctx.fillStyle = color; ctx.font = `${options.weight || 700} ${size}px ${FONT}`;
  ctx.textAlign = options.align || 'left'; ctx.textBaseline = 'middle';
  if (options.maxWidth) ctx.fillText(String(value), x, y, options.maxWidth);
  else ctx.fillText(String(value), x, y);
}
function crown(ctx, x, y, color) {
  ctx.save();
  ctx.fillStyle = color; ctx.strokeStyle = '#6f531866'; ctx.lineWidth = 2;
  ctx.lineJoin = 'round'; ctx.shadowColor = '#4c350055'; ctx.shadowBlur = 3; ctx.shadowOffsetY = 2;
  ctx.beginPath();
  ctx.moveTo(x + 5, y + 14); ctx.lineTo(x + 22, y + 34); ctx.lineTo(x + 40, y + 7);
  ctx.lineTo(x + 58, y + 34); ctx.lineTo(x + 78, y + 14);
  ctx.lineTo(x + 70, y + 48); ctx.lineTo(x + 13, y + 48); ctx.closePath();
  ctx.fill(); ctx.stroke();
  ctx.shadowColor = 'transparent';
  rounded(ctx, x + 12, y + 46, 59, 9, 3, color);
  line(ctx, x + 17, y + 47, x + 66, y + 47, '#6f531855', 1.5);
  for (const [cx, cy] of [[x + 5, y + 13], [x + 40, y + 6], [x + 78, y + 13]]) {
    ctx.beginPath(); ctx.arc(cx, cy, 4.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  }
  ctx.restore();
}
function tileColors(value, palette, customTilePalette) {
  if (!value) return [palette.empty, ''];
  const custom = customTilePalette?.[value];
  if (custom) return Array.isArray(custom) ? custom : [custom.background, custom.color];
  return defaultTilePalette[value] || (value >= 32768 ? ['#050505', '#ffffff'] : ['#f2b179', '#ffffff']);
}
const GAME_BOARD_GAP_TO_TILE = 14 / 107.5;
const GAME_TILE_RADIUS_RATIO = 3 / 107.5;
const GAME_BOARD_RADIUS_RATIO = 6 / 500;
function tileFontSize(value, size) {
  const digits = String(value).length;
  if (value >= 2 && value <= 64) return size * (48 / 107.5);
  if (digits <= 3) return size * (40 / 107.5);
  if (digits === 4) return size * (32 / 107.5);
  return size * (24 / 107.5);
}
function centeredTileLabel(ctx, value, centerX, centerY, size, color, maxWidth) {
  ctx.fillStyle = color; ctx.font = `700 ${size}px ${FONT}`; ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  const metrics = ctx.measureText(String(value));
  const baseline = centerY + ((metrics.actualBoundingBoxAscent || size * .72)
    - (metrics.actualBoundingBoxDescent || size * .18)) / 2;
  ctx.fillText(String(value), centerX, baseline, maxWidth);
}
function board(ctx, values, variant, slot, palette, customTilePalette) {
  const [rows, cols] = VARIANTS[variant] || VARIANTS['4x4'];
  const widthUnits = cols + (cols + 1) * GAME_BOARD_GAP_TO_TILE;
  const heightUnits = rows + (rows + 1) * GAME_BOARD_GAP_TO_TILE;
  const size = Math.min(slot.width / widthUnits, slot.height / heightUnits);
  const gap = size * GAME_BOARD_GAP_TO_TILE;
  const boardW = cols * size + (cols + 1) * gap;
  const boardH = rows * size + (rows + 1) * gap;
  const x = slot.x + (slot.width - boardW) / 2;
  const y = slot.y + (slot.height - boardH) / 2;
  rounded(ctx, x, y, boardW, boardH, Math.min(boardW, boardH) * GAME_BOARD_RADIUS_RATIO, palette.board);
  for (let index = 0; index < rows * cols; index++) {
    const value = Number(values?.[index]) || 0;
    const tx = x + gap + (index % cols) * (size + gap);
    const ty = y + gap + Math.floor(index / cols) * (size + gap);
    const [background, foreground] = tileColors(value, palette, customTilePalette);
    rounded(ctx, tx, ty, size, size, size * GAME_TILE_RADIUS_RATIO, background);
    if (value) {
      centeredTileLabel(ctx, value, tx + size / 2, ty + size / 2,
        tileFontSize(value, size), foreground, size * .94);
    }
  }
}
function statBox(ctx, x, accent, firstTitle, firstValue, secondTitle, secondValue, palette) {
  rounded(ctx, x + 4, 61, 331, 155, 19, palette.cardShadow);
  rounded(ctx, x, 54, 335, 155, 19, palette.stats);
  rounded(ctx, x, 54, 9, 155, 5, accent);
  line(ctx, x + 169, 63, x + 169, 199, palette.statsLine);
  label(ctx, firstTitle, x + 85, 94, 22, palette.statsText, { align: 'center' });
  label(ctx, firstValue, x + 85, 166, 33, palette.statsText, { align: 'center', maxWidth: 153 });
  label(ctx, secondTitle, x + 254, 94, 22, palette.statsText, { align: 'center' });
  label(ctx, secondValue, x + 254, 166, 33, palette.statsText, { align: 'center', maxWidth: 145 });
}
function featuredCard(ctx, entry, variant, palette, customTilePalette, copy) {
  const b = posterCardBounds(0);
  rounded(ctx, b.x, b.y + 8, b.width, b.height, 22, palette.goldShadow);
  rounded(ctx, b.x, b.y, b.width, b.height, 22, palette.goldEdge);
  rounded(ctx, b.x + 14, b.y + 13, b.width - 28, b.height - 27, 14, palette.card);
  rounded(ctx, 82, 270, 251, 91, 18, palette.goldShadow);
  rounded(ctx, 82, 269, 251, 88, 18, palette.gold);
  crown(ctx, 103, 289, palette.crown);
  label(ctx, 'B1', 223, 317, 53, palette.crown);
  label(ctx, copy.finalScore, 94, 429, 27, palette.text);
  label(ctx, formatNumber(entry.score), 94, 492, 58, palette.text, { maxWidth: 445 });
  [[copy.boardSum, formatNumber(boardSum(entry.board))], [copy.singleRating, formatRating(entry.single_rating)], [copy.fourRate, fourRate(entry)]]
    .forEach(([key, value], index) => {
      const y = 563 + index * 50;
      label(ctx, key, 94, y, 26, palette.muted);
      label(ctx, value, 370, y, 28, palette.text, { align: 'right', maxWidth: 225 });
    });
  line(ctx, 94, 695, 522, 695, palette.cardLine, 2);
  label(ctx, dateText(entry.ended), 94, 724, 23, palette.muted, { weight: 400 });
  board(ctx, entry.board, variant, { x: 614, y: 269, width: 528, height: 474 }, palette, customTilePalette);
}
function smallCard(ctx, entry, index, variant, palette, customTilePalette, copy) {
  const b = posterCardBounds(index);
  rounded(ctx, b.x + 2, b.y + 6, b.width, b.height, 16, palette.cardShadow);
  rounded(ctx, b.x, b.y, b.width, b.height, 16, palette.cardLine);
  rounded(ctx, b.x + 2, b.y + 2, b.width - 4, b.height - 4, 15, palette.card);
  rounded(ctx, b.x + 14, b.y + 13, 94, 63, 15, palette.cardShadow);
  const rankColor = index === 1 ? palette.silver : index === 2 ? palette.bronze : palette.rank;
  rounded(ctx, b.x + 14, b.y + 11, 94, 61, 15, rankColor);
  label(ctx, `B${index + 1}`, b.x + 61, b.y + 45, 38, '#ffffff', { align: 'center', maxWidth: 84 });
  if (entry) {
    label(ctx, copy.finalScore, b.x + 126, b.y + 39, 19, palette.muted);
    label(ctx, formatNumber(entry.score), b.x + 126, b.y + 66, 25, palette.text, { maxWidth: 165 });
    [[copy.boardSum, formatNumber(boardSum(entry.board))], [copy.singleRating, formatRating(entry.single_rating)], [copy.fourRate, fourRate(entry)]]
      .forEach(([key, value], i) => {
        const y = b.y + 111 + i * 27;
        label(ctx, key, b.x + 22, y, 18, palette.muted);
        label(ctx, value, b.x + 268, y, 17, i === 0 ? palette.text : palette.muted,
          { align: 'right', maxWidth: 112 });
      });
    line(ctx, b.x + 21, b.y + 197, b.x + 270, b.y + 197, palette.cardLine, 1.5);
    label(ctx, dateText(entry.ended), b.x + 22, b.y + 218, 17, palette.muted, { weight: 400 });
    board(ctx, entry.board, variant, { x: b.x + 309, y: b.y + 11, width: 205, height: 211 }, palette, customTilePalette);
  } else {
    label(ctx, copy.empty, b.x + 132, b.y + 118, 24, palette.muted);
  }
}
function closingCard(ctx, palette, copy) {
  const x = 618, y = 1764, w = 530, h = 236;
  rounded(ctx, x + 2, y + 6, w, h, 17, palette.cardShadow);
  rounded(ctx, x, y, w, h, 17, palette.cardLine);
  rounded(ctx, x + 2, y + 2, w - 4, h - 4, 16, palette.quote);
  label(ctx, copy.quote1, x + w / 2, y + 64, 29, palette.muted, { align: 'center' });
  label(ctx, copy.quote2, x + w / 2, y + 111, 24, palette.muted, { align: 'center', maxWidth: 470 });
  line(ctx, x + w / 2 - 30, y + 146, x + w / 2 + 30, y + 146, palette.muted, 2);
  label(ctx, '2 0 4 8 · M O R E  T H A N  A  G A M E', x + w / 2, y + 187, 12,
    palette.muted, { align: 'center', maxWidth: 445 });
}

function landscapeStatBox(ctx, x, accent, firstTitle, firstValue, secondTitle, secondValue, palette) {
  const y = 43, w = 354, h = 132;
  rounded(ctx, x + 4, y + 5, w, h, 16, palette.cardShadow);
  rounded(ctx, x, y, w, h, 16, palette.stats);
  rounded(ctx, x, y, 8, h, 4, accent);
  line(ctx, x + w / 2, y + 14, x + w / 2, y + h - 14, palette.statsLine);
  label(ctx, firstTitle, x + w * .25, y + 34, 16, palette.statsText, { align: 'center' });
  label(ctx, firstValue, x + w * .25, y + 91, 29, palette.statsText, { align: 'center', maxWidth: w * .43 });
  label(ctx, secondTitle, x + w * .75, y + 34, 16, palette.statsText, { align: 'center' });
  label(ctx, secondValue, x + w * .75, y + 91, 29, palette.statsText, { align: 'center', maxWidth: w * .4 });
}

function landscapeHeader(ctx, { name, userId, variant, pbScore, pbRank, rating, raRank, entries, palette, copy }) {
  rounded(ctx, 32, 29, 1984, 178, 18, palette.cardShadow);
  rounded(ctx, 32, 24, 1984, 178, 18, palette.header);
  label(ctx, name || copy.player, 64, 73, 42, palette.headerText, { maxWidth: 500 });
  label(ctx, `${variant.replace('x', '×')} Best 10`, 64, 129, 28, palette.headerText, { maxWidth: 380 });
  label(ctx, `ID ${userId || '—'}`, 64, 168, 17, palette.headerText, { weight: 400, maxWidth: 380 });
  landscapeStatBox(ctx, 1234, palette.orange, 'PB', formatNumber(pbScore ?? entries[0]?.score),
    copy.pbRank, pbRank == null ? '—' : `#${pbRank}`, palette);
  landscapeStatBox(ctx, 1614, palette.mint, 'Rating', formatRating(rating),
    copy.raRank, raRank == null ? '—' : `#${raRank}`, palette);
}

function landscapeFeaturedCard(ctx, entry, variant, palette, customTilePalette, copy) {
  const b = posterCardBounds(0, 'landscape');
  rounded(ctx, b.x, b.y + 7, b.width, b.height, 20, palette.goldShadow);
  rounded(ctx, b.x, b.y, b.width, b.height, 20, palette.goldEdge);
  rounded(ctx, b.x + 12, b.y + 12, b.width - 24, b.height - 25, 13, palette.card);
  rounded(ctx, b.x + 28, b.y + 27, 184, 75, 16, palette.goldShadow);
  rounded(ctx, b.x + 28, b.y + 25, 184, 73, 16, palette.gold);
  crown(ctx, b.x + 41, b.y + 35, palette.crown);
  label(ctx, 'B1', b.x + 141, b.y + 63, 43, palette.crown, { align: 'center' });
  label(ctx, copy.finalScore, b.x + 249, b.y + 45, 19, palette.muted);
  label(ctx, formatNumber(entry.score), b.x + 249, b.y + 81, 39, palette.text, { maxWidth: 485 });
  board(ctx, entry.board, variant, { x: b.x + 60, y: b.y + 128, width: 660, height: 624 }, palette, customTilePalette);
  [[copy.boardSum, formatNumber(boardSum(entry.board))], [copy.singleRating, formatRating(entry.single_rating)], [copy.fourRate, fourRate(entry)]]
    .forEach(([key, value], index) => {
      const x = b.x + 60 + index * 224;
      label(ctx, key, x, b.y + 785, 17, palette.muted, { maxWidth: 205 });
      label(ctx, value, x, b.y + 827, 26, palette.text, { maxWidth: 205 });
    });
  line(ctx, b.x + 60, b.y + 865, b.x + b.width - 60, b.y + 865, palette.cardLine, 2);
  label(ctx, dateText(entry.ended), b.x + 60, b.y + 901, 18, palette.muted, { weight: 400 });
  label(ctx, copy.footer, b.x + b.width - 60, b.y + 901, 13, palette.muted,
    { align: 'right', weight: 400, maxWidth: 420 });
}

function landscapePodiumCard(ctx, entry, index, variant, palette, customTilePalette, copy) {
  const b = posterCardBounds(index, 'landscape');
  rounded(ctx, b.x + 2, b.y + 5, b.width, b.height, 14, palette.cardShadow);
  rounded(ctx, b.x, b.y, b.width, b.height, 14, palette.cardLine);
  rounded(ctx, b.x + 2, b.y + 2, b.width - 4, b.height - 4, 13, palette.card);
  const rankColor = index === 1 ? palette.silver : index === 2 ? palette.bronze : palette.rank;
  rounded(ctx, b.x + 15, b.y + 14, 76, 54, 11, rankColor);
  label(ctx, `B${index + 1}`, b.x + 53, b.y + 41, 29, '#ffffff', { align: 'center', maxWidth: 69 });
  if (!entry) {
    label(ctx, copy.empty, b.x + b.width / 2, b.y + b.height / 2, 19, palette.muted, { align: 'center' });
    return;
  }
  label(ctx, copy.finalScore, b.x + 108, b.y + 28, 14, palette.muted, { maxWidth: 190 });
  label(ctx, formatNumber(entry.score), b.x + 108, b.y + 57, 23, palette.text, { maxWidth: 205 });
  [[copy.boardSum, formatNumber(boardSum(entry.board))], [copy.singleRating, formatRating(entry.single_rating)], [copy.fourRate, fourRate(entry)]]
    .forEach(([key, value], i) => {
      const y = b.y + 129 + i * 39;
      label(ctx, key, b.x + 24, y, 15, palette.muted, { maxWidth: 135 });
      label(ctx, value, b.x + 304, y, 16, i === 0 ? palette.text : palette.muted,
        { align: 'right', maxWidth: 126 });
    });
  line(ctx, b.x + 24, b.y + 274, b.x + 306, b.y + 274, palette.cardLine, 1.2);
  label(ctx, dateText(entry.ended), b.x + 24, b.y + 302, 14, palette.muted, { weight: 400, maxWidth: 275 });
  board(ctx, entry.board, variant, { x: b.x + 332, y: b.y + 82, width: 226, height: 226 }, palette, customTilePalette);
}

function landscapeCompactCard(ctx, entry, index, variant, palette, customTilePalette, copy) {
  const b = posterCardBounds(index, 'landscape');
  rounded(ctx, b.x + 2, b.y + 5, b.width, b.height, 13, palette.cardShadow);
  rounded(ctx, b.x, b.y, b.width, b.height, 13, palette.cardLine);
  rounded(ctx, b.x + 2, b.y + 2, b.width - 4, b.height - 4, 12, palette.card);
  rounded(ctx, b.x + 13, b.y + 13, 58, 43, 10, palette.rank);
  label(ctx, `B${index + 1}`, b.x + 42, b.y + 35, 22, '#ffffff', { align: 'center', maxWidth: 53 });
  if (!entry) {
    label(ctx, copy.empty, b.x + b.width / 2, b.y + b.height / 2, 17, palette.muted, { align: 'center' });
    return;
  }
  label(ctx, copy.finalScore, b.x + 81, b.y + 24, 12, palette.muted, { maxWidth: 90 });
  label(ctx, formatNumber(entry.score), b.x + 81, b.y + 47, 18, palette.text, { maxWidth: 130 });
  [[copy.boardSum, formatNumber(boardSum(entry.board))], [copy.singleRating, formatRating(entry.single_rating)], [copy.fourRate, fourRate(entry)]]
    .forEach(([key, value], i) => {
      const y = b.y + 108 + i * 34;
      label(ctx, key, b.x + 17, y, 12, palette.muted, { maxWidth: 78 });
      label(ctx, value, b.x + 140, y, 13, i === 0 ? palette.text : palette.muted,
        { align: 'right', maxWidth: 65 });
    });
  line(ctx, b.x + 17, b.y + 253, b.x + 142, b.y + 253, palette.cardLine, 1.1);
  label(ctx, dateText(entry.ended), b.x + 17, b.y + 276, 11.5, palette.muted, { weight: 400, maxWidth: 128 });
  board(ctx, entry.board, variant, { x: b.x + 154, y: b.y + 82, width: 117, height: 163 }, palette, customTilePalette);
}

function landscapeClosingCard(ctx, palette, copy) {
  const x = 1730, y = 887, w = 286, h = 295;
  rounded(ctx, x + 2, y + 5, w, h, 13, palette.cardShadow);
  rounded(ctx, x, y, w, h, 13, palette.cardLine);
  rounded(ctx, x + 2, y + 2, w - 4, h - 4, 12, palette.quote);
  label(ctx, copy.quote1, x + w / 2, y + 82, 24, palette.muted, { align: 'center' });
  label(ctx, copy.quote2, x + w / 2, y + 126, 17, palette.muted, { align: 'center', maxWidth: 250 });
  line(ctx, x + w / 2 - 28, y + 169, x + w / 2 + 28, y + 169, palette.muted, 1.5);
  label(ctx, '2 0 4 8', x + w / 2, y + 211, 16, palette.muted, { align: 'center' });
  label(ctx, 'MORE THAN A GAME', x + w / 2, y + 244, 10.5, palette.muted, { align: 'center' });
}

function drawLandscape(ctx, options) {
  const { name, userId, variant, entries, pbScore, pbRank, rating, raRank,
    palette, copy, tilePalette } = options;
  ctx.fillStyle = palette.page;
  ctx.fillRect(0, 0, LANDSCAPE_POSTER_WIDTH, LANDSCAPE_POSTER_HEIGHT);
  landscapeHeader(ctx, { name, userId, variant, pbScore, pbRank, rating, raRank, entries, palette, copy });
  if (entries.length) landscapeFeaturedCard(ctx, entries[0], variant, palette, tilePalette, copy);
  for (let index = 1; index <= 2; index++)
    landscapePodiumCard(ctx, entries[index], index, variant, palette, tilePalette, copy);
  for (let index = 3; index <= 9; index++)
    landscapeCompactCard(ctx, entries[index], index, variant, palette, tilePalette, copy);
  landscapeClosingCard(ctx, palette, copy);
}

export async function drawBestTenPoster({ canvas = document.createElement('canvas'), name, userId, variant,
  entries, pbScore, pbRank, rating, raRank, dark = false, language = 'zh', tilePalette = null,
  layout = 'portrait' }) {
  await document.fonts.ready;
  const landscape = layout === 'landscape';
  canvas.width = landscape ? 2700 : 1600;
  canvas.height = landscape ? 1600 : 2700;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('canvas_unavailable');
  ctx.scale(canvas.width / (landscape ? LANDSCAPE_POSTER_WIDTH : POSTER_WIDTH),
    canvas.height / (landscape ? LANDSCAPE_POSTER_HEIGHT : POSTER_HEIGHT));
  const palette = dark ? palettes.dark : palettes.light;
  const copy = COPY[language] || COPY.zh;
  if (landscape) {
    drawLandscape(ctx, { name, userId, variant, entries, pbScore, pbRank, rating, raRank,
      palette, copy, tilePalette });
    return canvas;
  }
  ctx.fillStyle = palette.page; ctx.fillRect(0, 0, POSTER_WIDTH, POSTER_HEIGHT);
  rounded(ctx, 47, 31, 1121, 208, 19, palette.cardShadow);
  rounded(ctx, 47, 25, 1121, 208, 19, palette.header);
  label(ctx, name || copy.player, 84, 88, 53, palette.headerText, { maxWidth: 355 });
  label(ctx, `${variant.replace('x', '×')} Best 10`, 84, 171, 36, palette.headerText, { maxWidth: 350 });
  label(ctx, `ID ${userId || '—'}`, 84, 209, 24, palette.headerText, { weight: 400 });
  statBox(ctx, 461, palette.orange, 'PB', formatNumber(pbScore ?? entries[0]?.score),
    copy.pbRank, pbRank == null ? '—' : `#${pbRank}`, palette);
  statBox(ctx, 812, palette.mint, 'Rating', formatRating(rating),
    copy.raRank, raRank == null ? '—' : `#${raRank}`, palette);
  if (entries.length) featuredCard(ctx, entries[0], variant, palette, tilePalette, copy);
  for (let index = 1; index <= 9; index++) smallCard(ctx, entries[index], index, variant, palette, tilePalette, copy);
  closingCard(ctx, palette, copy);
  label(ctx, copy.footer,
  1148, 2025, 15.6, palette.muted, { align: 'right', weight: 400, maxWidth: 520 });
  return canvas;
}
