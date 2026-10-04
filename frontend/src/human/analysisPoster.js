// Single-game analysis poster. The same canvas is used for preview and PNG export.
import { goalTargetLabel } from '../utils/goalTarget.js';
export const ANALYSIS_POSTER_WIDTH = 2360;
export const ANALYSIS_POSTER_HEIGHT = 1640;
const W = 1180, H = 820;
const FONT = '"Clear Sans", "Microsoft YaHei", Arial, sans-serif';
const BOARD_SHAPES = { '4x4': [4, 4], '3x4': [3, 4], '2x4': [2, 4], '3x3': [3, 3] };
const COUNTS = ['Perfect!', 'Excellent!', 'Nice try!', 'Not bad!', 'Mistake!', 'Blunder!', 'Terrible!'];
const COUNT_COLORS = ['#2e7d32', '#7cb342', '#c0ca33', '#fb8c00', '#f4511e', '#e53935', '#b71c1c'];
const ENDGAME_TIERS = { 16384: '16K', 32768: '32K', 65536: '65K' };
const TILE_COLORS = {
  2: ['#f9f1e8', '#746b63'], 4: ['#f7ead5', '#746b63'],
  8: ['#f2b179', '#fff'], 16: ['#f2b179', '#fff'],
  32: ['#f67c5f', '#fff'], 64: ['#f65e3b', '#fff'],
  128: ['#f1d271', '#fff'], 256: ['#f1d271', '#fff'],
  512: ['#f3ca4e', '#fff'], 1024: ['#f3c43c', '#fff'],
  2048: ['#f6ca2d', '#fff'], 4096: ['#6500a4', '#fff'],
  8192: ['#8700c3', '#fff'], 16384: ['#4c0073', '#fff'],
  32768: ['#080808', '#fff'], 65536: ['#050505', '#fff'],
};
const COPY = {
  zh: { analysis: '对局分析', finalBoard: '终局盘面', score: '终局分数',
    accuracy: '平均单步准确率', perfect: 'PERFECT', elapsed: '对局用时', sum: '终局总和',
    ratedMoves: '评价步数', fit: '平均吻合度', combo: 'MAX COMBO', eval: '评价统计', best: 'NEW BEST',
    previous: '前 PB', personal: '个人', endgame: '残局',
    stages: '残局数', moves: '残局总步数', played: '对局日期',
    pending: '待评级', site: '来本站对局、分析并生成你的展示图', sample: '样式示例 · 非真实成绩' },
  en: { analysis: 'GAME ANALYSIS', finalBoard: 'FINAL BOARD', score: 'FINAL SCORE',
    accuracy: 'GEOMETRIC ACCURACY', perfect: 'PERFECT', elapsed: 'GAME TIME', sum: 'FINAL BOARD SUM',
    ratedMoves: 'EVALUATED MOVES', fit: 'AVERAGE FIT', combo: 'MAX COMBO', eval: 'EVALUATION', best: 'NEW BEST',
    previous: 'PREVIOUS PB', personal: 'PERSONAL', endgame: 'ENDGAME',
    stages: 'ENDGAMES', moves: 'ENDGAME MOVES', played: 'PLAYED',
    pending: 'UNRATED', site: 'Play, analyze and make your own result card', sample: 'STYLE PREVIEW · SAMPLE DATA' },
};

function roundedRectPath(ctx, x, y, width, height, radius) {
  const r = Math.max(0, Math.min(Number(radius) || 0, Math.abs(width) / 2, Math.abs(height) / 2));
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.arcTo(x + width, y, x + width, y + r, r);
  ctx.lineTo(x + width, y + height - r);
  ctx.arcTo(x + width, y + height, x + width - r, y + height, r);
  ctx.lineTo(x + r, y + height);
  ctx.arcTo(x, y + height, x, y + height - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}
function fillRound(ctx, x, y, width, height, radius, color) {
  ctx.fillStyle = color;
  ctx.beginPath(); roundedRectPath(ctx, x, y, width, height, radius); ctx.fill();
}
function text(ctx, value, x, y, size, color, options = {}) {
  ctx.font = `${options.weight || 700} ${size}px ${FONT}`;
  ctx.fillStyle = color;
  ctx.textAlign = options.align || 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(String(value), x, y, options.maxWidth);
}
function metricText(ctx, value, x, y, size, maxWidth) {
  ctx.save();
  ctx.font = `700 ${size}px "Trebuchet MS", ${FONT}`;
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  ctx.lineJoin = 'round'; ctx.lineWidth = 1.1;
  ctx.strokeStyle = '#725b7b99';
  ctx.shadowColor = '#d7a9ed66'; ctx.shadowBlur = 9; ctx.shadowOffsetY = 2;
  ctx.strokeText(String(value), x, y, maxWidth);
  const gradient = ctx.createLinearGradient(0, y - size * .55, 0, y + size * .55);
  gradient.addColorStop(0, '#fffdf7');
  gradient.addColorStop(.52, '#f4e6ff');
  gradient.addColorStop(1, '#cda9dd');
  ctx.fillStyle = gradient;
  ctx.fillText(String(value), x, y, maxWidth);
  ctx.restore();
}
function cutPanel(ctx, x, y, width, height, color, inset = 18) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(x + inset, y);
  ctx.lineTo(x + width, y);
  ctx.lineTo(x + width - inset, y + height);
  ctx.lineTo(x, y + height);
  ctx.closePath(); ctx.fill();
}
function line(ctx, x1, y1, x2, y2, color, width = 1) {
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
}
function evaluationBar(ctx, counts, x, y, width, height) {
  const values = COUNTS.map(label => Math.max(0, Number(counts?.[label]) || 0));
  const total = values.reduce((sum, value) => sum + value, 0);
  fillRound(ctx, x, y, width, height, height / 2, '#302b35');
  if (total > 0) {
    ctx.save(); ctx.beginPath(); roundedRectPath(ctx, x, y, width, height, height / 2); ctx.clip();
    let cursor = x;
    let lastPositive = values.length - 1;
    while (lastPositive > 0 && !values[lastPositive]) lastPositive -= 1;
    values.forEach((value, index) => {
      if (!value) return;
      const segmentWidth = index === lastPositive ? x + width - cursor : width * value / total;
      ctx.fillStyle = COUNT_COLORS[index];
      ctx.fillRect(cursor, y, segmentWidth, height);
      cursor += segmentWidth;
    });
    ctx.restore();
  }
  ctx.strokeStyle = '#d6c5dc55'; ctx.lineWidth = .75;
  ctx.beginPath(); roundedRectPath(ctx, x, y, width, height, height / 2); ctx.stroke();
}
function tilePaint(value, palette) {
  const custom = palette?.[value];
  if (custom) return Array.isArray(custom) ? custom : [custom.background, custom.color];
  return TILE_COLORS[value] || (value >= 32768 ? ['#050505', '#fff'] : ['#f2b179', '#fff']);
}
// Matches getTileLabelStyle() on the Play board at the default font scale.
function tileFontSize(value, scale) {
  if (value <= 64) return 48 * scale;
  const digits = String(value).length;
  return (digits > 4 ? 24 : digits > 3 ? 32 : 40) * scale;
}
function centeredTileText(ctx, value, x, y, cell, color, scale) {
  let size = tileFontSize(value, scale);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = color;
  ctx.font = `700 ${size}px ${FONT}`;
  const label = String(value);
  const measured = ctx.measureText(label);
  if (measured.width > cell - 8) {
    size *= (cell - 8) / measured.width;
    ctx.font = `700 ${size}px ${FONT}`;
  }
  const ink = ctx.measureText(label);
  const centerOffset = Number.isFinite(ink.actualBoundingBoxAscent)
    ? (ink.actualBoundingBoxAscent - ink.actualBoundingBoxDescent) / 2 : size * .35;
  ctx.fillText(label, x + cell / 2, y + cell / 2 + centerOffset);
}
function boardGeometry(variant, frame) {
  const [rows, cols] = BOARD_SHAPES[variant] || BOARD_SHAPES['4x4'];
  const gap = variant === '3x3' ? Math.min(frame.width, frame.height) * .038 : 8;
  const cell = Math.min((frame.width - gap * (cols + 1)) / cols,
    (frame.height - gap * (rows + 1)) / rows);
  const width = cols * cell + (cols + 1) * gap;
  const height = rows * cell + (rows + 1) * gap;
  return { rows, cols, gap, cell, width, height,
    x: frame.x + (frame.width - width) / 2, y: frame.y + (frame.height - height) / 2 };
}
function drawBoard(ctx, values, variant, frame, palette, fontScale) {
  const b = boardGeometry(variant, frame);
  fillRound(ctx, b.x, b.y, b.width, b.height, 14, '#978980');
  for (let i = 0; i < b.rows * b.cols; i++) {
    const value = Number(values?.[i]) || 0;
    const x = b.x + b.gap + (i % b.cols) * (b.cell + b.gap);
    const y = b.y + b.gap + Math.floor(i / b.cols) * (b.cell + b.gap);
    const [background, foreground] = value ? tilePaint(value, palette) : ['#b5a69b', ''];
    fillRound(ctx, x, y, b.cell, b.cell, 3, background);
    if (value) {
      centeredTileText(ctx, value, x, y, b.cell, foreground,
        fontScale * (variant === '3x3' ? b.cell / 96.25 : 1));
    }
  }
}
function ambient(ctx, board, variant, palette, fontScale) {
  ctx.fillStyle = '#18171f'; ctx.fillRect(0, 0, W, H);
  const background = document.createElement('canvas');
  background.width = 560; background.height = 560;
  const bg = background.getContext('2d');
  if (bg) {
    drawBoard(bg, board, variant, { x: 0, y: 0, width: 560, height: 560 }, palette, fontScale);
    ctx.save();
    ctx.globalAlpha = .7;
    ctx.filter = 'blur(85px)';
    ctx.drawImage(background, -110, -130, 1400, 1100);
    ctx.restore();
  }
  // A deterministic fallback gives the card depth even without Canvas filters.
  const glow = ctx.createRadialGradient(250, 520, 20, 300, 500, 660);
  glow.addColorStop(0, '#9f578ca0'); glow.addColorStop(1, '#211b3200');
  ctx.fillStyle = glow; ctx.fillRect(0, 0, W, H);
  const violet = ctx.createRadialGradient(1010, 150, 10, 900, 120, 520);
  violet.addColorStop(0, '#77509c65'); violet.addColorStop(1, '#17122100');
  ctx.fillStyle = violet; ctx.fillRect(0, 0, W, H);
  const shade = ctx.createLinearGradient(0, 0, W, H);
  shade.addColorStop(0, '#0c0c14b8'); shade.addColorStop(.48, '#19151cc4');
  shade.addColorStop(1, '#0b0c16ed');
  ctx.fillStyle = shade; ctx.fillRect(0, 0, W, H);
}
function avatar(ctx, name, image) {
  const x = 1045, y = 43, size = 75;
  ctx.save(); ctx.beginPath(); roundedRectPath(ctx, x, y, size, size, 9); ctx.clip();
  if (image?.complete && image.naturalWidth) {
    const scale = Math.max(size / image.naturalWidth, size / image.naturalHeight);
    ctx.drawImage(image, x + (size - image.naturalWidth * scale) / 2,
      y + (size - image.naturalHeight * scale) / 2,
      image.naturalWidth * scale, image.naturalHeight * scale);
  } else {
    const gradient = ctx.createLinearGradient(x, y, x + size, y + size);
    gradient.addColorStop(0, '#a879a3'); gradient.addColorStop(1, '#443051');
    ctx.fillStyle = gradient; ctx.fillRect(x, y, size, size);
    text(ctx, (name || 'P').slice(0, 1).toUpperCase(), x + size / 2, y + size / 2,
      43, '#fff', { align: 'center' });
  }
  ctx.restore();
}
function formatDate(seconds) {
  if (!Number.isFinite(Number(seconds))) return '—';
  const date = new Date(Number(seconds) * 1000);
  if (Number.isNaN(date.getTime())) return '—';
  const parts = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Shanghai',
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
    hour12: false }).formatToParts(date);
  const part = type => parts.find(item => item.type === type)?.value || '00';
  return `${part('year')}-${part('month')}-${part('day')} ${part('hour')}:${part('minute')}`;
}
const number = value => Number.isFinite(Number(value)) ? new Intl.NumberFormat('en-US').format(Number(value)) : '—';
export const analysisPercent = (value, digits = 2) => value != null && Number.isFinite(Number(value))
  ? `${(Number(value) * 100).toFixed(digits)}%` : '—';
const percent = analysisPercent;
export function analysisDuration(ms) {
  if (ms == null || !Number.isFinite(Number(ms)) || Number(ms) <= 0) return '—';
  const seconds = Math.floor(Number(ms) / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

const GRADE_STYLES = {
  X: { light: '#f0ffff', mid: '#85e5ed', dark: '#318caa', glow: '#9ef5ff' },
  SSS: { light: '#fff3c9', mid: '#eac77e', dark: '#b17937', glow: '#f0c073' },
  SS: { light: '#f8e9ff', mid: '#d8b4ed', dark: '#8f5db2', glow: '#d9a7ed' },
  S: { light: '#eedcff', mid: '#b78bef', dark: '#7147a9', glow: '#a76de5' },
  A: { light: '#d9fbff', mid: '#80d4e0', dark: '#317f99', glow: '#7acddd' },
  B: { light: '#e4fbdc', mid: '#9bc885', dark: '#507d52', glow: '#91bf85' },
  C: { light: '#faf5ea', mid: '#c4b8a8', dark: '#746c67', glow: '#c1b4a4' },
  D: { light: '#ffe9d1', mid: '#d4a878', dark: '#8b5a43', glow: '#d3a177' },
  E: { light: '#ffdfd9', mid: '#dc8d81', dark: '#9f4d59', glow: '#d88580' },
  F: { light: '#ffdadd', mid: '#c57383', dark: '#80384e', glow: '#c46e80' },
};
function drawGrade(ctx, rawGrade, copy) {
  const grade = String(rawGrade || '').toUpperCase();
  const style = GRADE_STYLES[grade];
  if (!style) {
    text(ctx, '—', 1034, 289, 76, '#beb0c0', { align: 'center', weight: 400 });
    text(ctx, copy.pending, 1034, 358, 14, '#cfb4d5', { align: 'center', weight: 500 });
    return;
  }
  const centerX = 1034;
  const size = grade.length === 3 ? 76 : grade.length === 2 ? 92 : 106;
  const font = `italic 700 ${size}px "Trebuchet MS", "Clear Sans", Arial, sans-serif`;
  ctx.save();
  const halo = ctx.createRadialGradient(centerX, 285, 4, centerX, 285, 115);
  halo.addColorStop(0, `${style.glow}45`); halo.addColorStop(1, `${style.glow}00`);
  ctx.fillStyle = halo; ctx.fillRect(956, 223, 160, 139);
  ctx.font = font;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  const width = ctx.measureText(grade).width;
  const available = 155;
  if (width > available) {
    ctx.translate(centerX, 0);
    ctx.scale(available / width, 1);
    ctx.translate(-centerX, 0);
  }
  ctx.lineJoin = 'round'; ctx.lineWidth = 1.6; ctx.strokeStyle = style.dark;
  ctx.shadowColor = style.glow; ctx.shadowBlur = grade === 'X' || grade.startsWith('S') ? 16 : 6;
  ctx.strokeText(grade, centerX, 285);
  const gradient = ctx.createLinearGradient(0, 237, 0, 329);
  gradient.addColorStop(0, style.light);
  gradient.addColorStop(.55, style.mid);
  gradient.addColorStop(1, style.dark);
  ctx.fillStyle = gradient; ctx.fillText(grade, centerX, 285);
  ctx.restore();
  line(ctx, 976, 330, 1092, 330, style.mid, 2);
  text(ctx, 'GRADE', centerX, 350, 13, style.light, { align: 'center', weight: 700 });
}

export async function drawAnalysisPoster({ canvas = document.createElement('canvas'),
  data, qrImage = null, qrBackdropImage = null, avatarImage = null,
  tilePalette = null, tileFontScale = null,
  language = 'zh' }) {
  await document.fonts?.ready;
  canvas.width = ANALYSIS_POSTER_WIDTH; canvas.height = ANALYSIS_POSTER_HEIGHT;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('canvas_unavailable');
  ctx.setTransform(2, 0, 0, 2, 0, 0);
  const copy = COPY[language] || COPY.zh;
  const run = data.run || {}, aggregate = data.aggregate || {};
  const name = data.name || 'Player', variant = run.variant || '4x4';
  const is3x3 = variant === '3x3';
  const configuredScale = Number.parseFloat(getComputedStyle(document.documentElement)
    .getPropertyValue('--tile-font-scale'));
  const fontScale = Number.isFinite(Number(tileFontScale)) && Number(tileFontScale) > 0
    ? Number(tileFontScale) : Number.isFinite(configuredScale) && configuredScale > 0
      ? configuredScale : 1;
  ambient(ctx, run.board, variant, tilePalette, fontScale);
  if (qrBackdropImage?.complete && qrBackdropImage.naturalWidth) {
    ctx.drawImage(qrBackdropImage, 788, 570, 520, 520);
  }

  // Header: small identity on the left, player identity on the right.
  fillRound(ctx, 52, 43, 5, 63, 2, '#cf9fef');
  text(ctx, '2048', 69, 66, 30, '#fff', { weight: 800 });
  text(ctx, `PLAY  /  ${copy.analysis}`, 69, 99, 12, '#d7cdda', { weight: 500 });
  text(ctx, name, 1024, 70, 28, '#fff', { align: 'right', maxWidth: 420 });
  const formation = is3x3 ? goalTargetLabel(data.target, language)
    : `${data.pattern || '—'}-${goalTargetLabel(data.target, language) || '—'}`;
  text(ctx, `${variant.replace('x', '×')}  ·  ${formation}`,
    1024, 104, 15, '#e1cfec', { align: 'right', weight: 500 });
  avatar(ctx, name, avatarImage);
  line(ctx, 52, 147, 1128, 147, '#d5b6dc66');

  // Final board is the visual anchor, and also drives the blurred background.
  fillRound(ctx, 55, 183, 508, 500, 8, '#15131bd9');
  ctx.strokeStyle = '#d1b1d482'; ctx.lineWidth = 1;
  ctx.strokeRect(69, 198, 480, 470);
  drawBoard(ctx, run.board, variant, { x: 92, y: 212, width: 425, height: 425 }, tilePalette, fontScale);
  text(ctx, copy.finalBoard, 86, 651, 14, '#e2d4e8', { weight: 500 });
  text(ctx, variant.replace('x', ' × '), 524, 651, 14, '#e2d4e8', { align: 'right', weight: 500 });

  // The score and grade share a single dominant panel, like a rhythm-game result.
  cutPanel(ctx, 592, 183, 535, 238, '#101016e8', 19);
  text(ctx, copy.score, 622, 224, 16, '#cbbdce', { weight: 500 });
  metricText(ctx, number(run.score), 622, 297, 51, 285);
  line(ctx, 932, 228, 932, 348, '#a98cac77');
  drawGrade(ctx, data.grade, copy);
  if (run.pb?.new_best) {
    line(ctx, 622, 367, 1085, 367, '#9b7d9d88');
    text(ctx, copy.best, 622, 392, 17, '#ffdaa0');
    text(ctx, `${copy.previous} ${number(run.pb.previous_best)}`, 756, 392, 15, '#cbbdce', { weight: 500 });
    text(ctx, `+${number(run.pb.delta)}`, 1085, 392, 19, '#ffdaa0', { align: 'right' });
  } else if (Number.isInteger(Number(run.personal_rank)) && Number(run.personal_rank) > 0) {
    line(ctx, 622, 367, 1085, 367, '#9b7d9d88');
    text(ctx, `${copy.personal}  #${number(run.personal_rank)}`, 622, 392, 18, '#e6c7f0', { weight: 700 });
  }

  cutPanel(ctx, 596, 438, 527, 110, '#141219df', 15);
  if (is3x3) {
    line(ctx, 786, 457, 786, 530, '#b8a9bb70');
    line(ctx, 949, 457, 949, 530, '#b8a9bb70');
    metricText(ctx, percent(aggregate.mean_single_step_accuracy, 5), 623, 477, 27, 155);
    text(ctx, copy.accuracy, 624, 521, 11, '#d0c4d4', { weight: 500, maxWidth: 156 });
    metricText(ctx, percent(aggregate.perfect_rate), 803, 477, 31, 134);
    text(ctx, copy.perfect, 804, 521, 12, '#d0c4d4', { weight: 500 });
    metricText(ctx, number(aggregate.max_combo), 966, 477, 36, 131);
    text(ctx, copy.combo, 967, 521, 12, '#d0c4d4', { weight: 500 });
  } else {
    line(ctx, 854, 457, 854, 530, '#b8a9bb70');
    metricText(ctx, number(aggregate.max_combo), 625, 477, 39, 200);
    text(ctx, copy.combo, 626, 521, 14, '#d0c4d4', { weight: 500 });
    metricText(ctx, percent(aggregate.mean_goodness_of_fit), 882, 477, 39, 215);
    const tier = ENDGAME_TIERS[Number(data.goalTile)];
    const tierLabel = tier ? (language === 'zh' ? `${tier}${copy.endgame}` : `${tier} ${copy.endgame}`) : '';
    const fitLabel = tierLabel ? `${tierLabel} · ${copy.fit}` : copy.fit;
    text(ctx, fitLabel, 883, 521, 12.5, '#d0c4d4', { weight: 500, maxWidth: 215 });
  }

  cutPanel(ctx, 601, 565, 518, 118, '#141219df', 15);
  text(ctx, copy.eval, 624, 587, 12, '#c6b7cc', { weight: 500 });
  const start = 619, slot = 69;
  COUNTS.forEach((label, index) => {
    const x = start + index * slot;
    text(ctx, number(aggregate.performance_counts?.[label] ?? 0), x, 625, 24,
      index === 0 ? '#ffdfaa' : '#fff', { maxWidth: 63 });
    text(ctx, label.replace('!', ''), x, 660, 10.5, '#c7bacb',
      { weight: 500, maxWidth: 65 });
  });
  evaluationBar(ctx, aggregate.performance_counts, 624, 674, 466, 5);

  // A restrained footer keeps the image attributable after it is shared.
  line(ctx, 55, 713, 1124, 713, '#d5b6dc66');
  if (is3x3) {
    text(ctx, `${copy.ratedMoves}  ${number(aggregate.evaluated_moves)}   /   ${copy.stages}  ${number(aggregate.stage_count)}`,
      60, 734, 13, '#efe7f0', { weight: 500 });
    text(ctx, `${copy.elapsed}  ${analysisDuration(aggregate.run_elapsed_ms)}   /   ${copy.sum}  ${number(aggregate.run_board_sum)}`,
      60, 758, 13, '#efe7f0', { weight: 500 });
  } else {
    text(ctx, `${copy.stages}  ${number(aggregate.stage_count ?? 0)}   /   ${copy.moves}  ${number(aggregate.total_moves ?? 0)}`,
      60, 743, 15, '#efe7f0', { weight: 500 });
  }
  text(ctx, `${copy.played}  ${formatDate(run.ended_at)}`, 60, 782, 14, '#cdbed0', { weight: 500 });
  text(ctx, 'play.2048tables.online', 790, 743, 20, '#fff', { align: 'right' });
  text(ctx, copy.site, 790, 779, 12.5, '#d8cbdc', { align: 'right', weight: 500, maxWidth: 650 });
  if (qrImage?.complete && qrImage.naturalWidth) {
    ctx.drawImage(qrImage, 999, 694, 118, 118);
    fillRound(ctx, 1047, 742, 22, 22, 3, '#f6ca2d');
    text(ctx, '2048', 1058, 753, 7.5, '#4d3341', { align: 'center', weight: 800 });
  } else text(ctx, 'QR', 1058, 753, 30, '#3b2649', { align: 'center' });
  if (data.sample) text(ctx, copy.sample, 791, 805, 10, '#dbc8df',
    { align: 'right', weight: 500 });
  return canvas;
}
