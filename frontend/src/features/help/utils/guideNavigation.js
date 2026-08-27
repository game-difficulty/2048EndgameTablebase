const GUIDE_HEX_LENGTH = 16;

export function normalizeGuideBoardHex(value, visibleRows = 4, visibleCols = 4) {
  const normalized = String(value || '')
    .trim()
    .replace(/^0x/i, '')
    .toLowerCase();

  if (!/^[0-9a-f]+$/.test(normalized) || normalized.length > GUIDE_HEX_LENGTH) {
    return null;
  }

  const rows = Number(visibleRows);
  const cols = Number(visibleCols);
  const visibleCellCount = Number.isInteger(rows) && Number.isInteger(cols)
    ? rows * cols
    : GUIDE_HEX_LENGTH;

  if (normalized.length < GUIDE_HEX_LENGTH && normalized.length !== visibleCellCount) {
    return null;
  }

  return normalized.padEnd(GUIDE_HEX_LENGTH, 'f');
}

export function createGuideTrainerJumpDetail(board, documentId = '') {
  const hex = normalizeGuideBoardHex(
    board?.hex,
    board?.visible_rows,
    board?.visible_cols,
  );

  if (!hex) {
    return null;
  }

  return {
    hex,
    boardId: String(board?.board_id || ''),
    sourceDocumentId: String(documentId || ''),
  };
}
