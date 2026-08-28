const GUIDE_HEX_LENGTH = 16;

const normalizePaddingDigit = (value, fallback = 'f') => {
  const normalized = String(value || '').trim().toLowerCase();
  return /^[0-9a-f]$/.test(normalized) ? normalized : fallback;
};

export function normalizeGuideBoardHex(
  value,
  visibleRows = 4,
  visibleCols = 4,
  padding = {},
) {
  const normalized = String(value || '')
    .trim()
    .replace(/^0x/i, '')
    .toLowerCase();

  if (!/^[0-9a-f]+$/.test(normalized) || normalized.length > GUIDE_HEX_LENGTH) {
    return null;
  }

  const rows = Number(visibleRows);
  const cols = Number(visibleCols);
  const dimensionsValid = Number.isInteger(rows)
    && Number.isInteger(cols)
    && rows >= 1
    && rows <= 4
    && cols >= 1
    && cols <= 4;
  const visibleCellCount = dimensionsValid ? rows * cols : GUIDE_HEX_LENGTH;

  if (normalized.length < GUIDE_HEX_LENGTH && normalized.length !== visibleCellCount) {
    return null;
  }
  if (normalized.length === GUIDE_HEX_LENGTH) {
    return normalized;
  }

  const rightPadding = normalizePaddingDigit(padding?.right);
  const bottomPadding = normalizePaddingDigit(padding?.bottom);
  const sourceRows = Array.from({ length: rows }, (_unused, row) => (
    normalized.slice(row * cols, (row + 1) * cols).padEnd(4, rightPadding).slice(0, 4)
  ));
  return sourceRows.join('').padEnd(GUIDE_HEX_LENGTH, bottomPadding).slice(0, GUIDE_HEX_LENGTH);
}

export function createGuideTrainerJumpDetail(board, documentId = '', trainerContext = {}) {
  const hex = normalizeGuideBoardHex(
    board?.hex,
    board?.visible_rows,
    board?.visible_cols,
    board?.padding,
  );
  if (!hex) {
    return null;
  }

  const detail = {
    hex,
    boardId: String(board?.board_id || ''),
    sourceDocumentId: String(documentId || ''),
  };
  const fullPattern = String(
    board?.trainer?.full_pattern || trainerContext?.full_pattern || '',
  ).trim();
  if (fullPattern) {
    detail.fullPattern = fullPattern;
  }
  return detail;
}
