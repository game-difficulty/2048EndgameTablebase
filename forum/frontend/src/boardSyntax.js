// Forum board syntax v1. Keep source text in PostgreSQL so edits are lossless.
// Same row-major exponent alphabet as Play practice, extended through 2^20.
export const BOARD_ALPHABET = "0123456789abcdefghijk";
export const BOARD_SHAPES = Object.freeze({
  "4x4": [4, 4],
  "3x4": [3, 4],
  "3x3": [3, 3],
  "2x4": [2, 4],
});
export const BOARD_EXAMPLE = "[[board:4x4:fedc/ba98/7654/3210|这里讨论下一步]]";
export const MAX_SYNTAX_BOARDS = 20;

export function decodeBoardSyntax(source) {
  const match =
    /^\[\[board:([^:\r\n]+):([^|\]\r\n]+)(?:\|([^\r\n]*))?\]\]$/i.exec(source);
  if (!match) return { error: "格式应为 [[board:变体:盘面编码|说明]]。" };
  const variant = match[1].trim().toLowerCase().replaceAll("×", "x");
  if (!Object.hasOwn(BOARD_SHAPES, variant))
    return { error: "支持的变体为 4x4、3x4、3x3、2x4。" };
  const [rows, cols] = BOARD_SHAPES[variant];
  let code = match[2].trim().replace(/^0x/i, "").toLowerCase();
  if (code.includes("/")) {
    const lines = code.split("/");
    if (lines.length !== rows || lines.some((line) => line.length !== cols))
      return { error: `分行编码需要 ${rows} 行，每行 ${cols} 个字符。` };
    code = lines.join("");
  }
  if (!/^[0-9a-k]+$/.test(code))
    return { error: "编码仅使用 0–9、a–k；0 表示空格，其他字符表示 2 的幂。" };
  if (code.length > rows * cols)
    return {
      error: `${variant} 最多包含 ${rows * cols} 个格子，不能截断多出的编码。`,
    };
  const caption = (match[3] || "").trim();
  if (caption.length > 500) return { error: "棋盘说明最多 500 字符。" };
  code = code.padStart(rows * cols, "0");
  return {
    board: {
      type: "board",
      rows,
      cols,
      cells: [...code].map((char) =>
        char === "0" ? 0 : 2 ** BOARD_ALPHABET.indexOf(char),
      ),
      caption,
    },
  };
}

export function encodeBoardSyntax(board) {
  const variant = `${board.rows}x${board.cols}`;
  if (
    !Object.hasOwn(BOARD_SHAPES, variant) ||
    board.cells.length !== board.rows * board.cols
  )
    return null;
  let code = "";
  for (const value of board.cells) {
    if (value === 0) {
      code += "0";
      continue;
    }
    const exponent = Math.log2(value);
    if (
      !Number.isInteger(exponent) ||
      exponent < 1 ||
      exponent >= BOARD_ALPHABET.length
    )
      return null;
    code += BOARD_ALPHABET[exponent];
  }
  // A caption containing a terminator/newline cannot be encoded losslessly.
  const caption = board.caption || "";
  if (/[\r\n]/.test(caption) || caption.includes("]]")) return null;
  return `[[board:${variant}:${code}${caption ? "|" + caption : ""}]]`;
}

export function parseBoardText(text, budget = MAX_SYNTAX_BOARDS) {
  const parts = [],
    diagnostics = [];
  let pending = "",
    pos = 0,
    boards = 0;
  const flush = () => {
    if (pending) {
      parts.push({ type: "paragraph", text: pending });
      pending = "";
    }
  };
  while (pos < text.length) {
    // Keep inline and fenced code literal, including an unfinished code span.
    if (text[pos] === "`" || text.slice(pos, pos + 3) === "~~~") {
      let endRun = pos;
      while (text[endRun] === text[pos]) endRun++;
      const marker = text.slice(pos, endRun);
      let close = text.indexOf(marker, endRun);
      while (
        close !== -1 &&
        (text[close - 1] === marker[0] ||
          text[close + marker.length] === marker[0])
      )
        close = text.indexOf(marker, close + marker.length);
      const end = close === -1 ? text.length : close + marker.length;
      pending += text.slice(pos, end);
      pos = end;
      continue;
    }
    if (
      text[pos] === "\\" &&
      /^\[\[(?:board|replay):/i.test(text.slice(pos + 1, pos + 10))
    ) {
      const close = text.indexOf("]]", pos + 9);
      const end = close === -1 ? text.length : close + 2;
      pending += text.slice(pos + 1, end);
      pos = end;
      continue;
    }
    if (!/^\[\[(?:board|replay):/i.test(text.slice(pos, pos + 9))) {
      pending += text[pos++];
      continue;
    }
    const close = text.indexOf("]]", pos + 8);
    if (close === -1) {
      diagnostics.push({
        offset: pos,
        message: "棋盘语法尚未结束，请补上 ]]。",
      });
      pending += text.slice(pos);
      break;
    }
    const raw = text.slice(pos, close + 2);
    // Don't consume a second valid token when the preceding token is malformed.
    const nested = raw.slice(8).toLowerCase().indexOf("[[board:");
    if (nested !== -1) {
      diagnostics.push({
        offset: pos,
        message: "前一段棋盘语法缺少结束符 ]]。",
      });
      pending += raw.slice(0, 8 + nested);
      pos += 8 + nested;
      continue;
    }
    const replay = /^\[\[replay:([0-9a-f-]{36})(?:@(\d{1,6}))?\]\]$/i.exec(raw);
    const result = replay
      ? {
          board: {
            type: "replay",
            id: replay[1].toLowerCase(),
            step: Math.min(200000, Number(replay[2] || 0)),
          },
        }
      : decodeBoardSyntax(raw);
    if (result.error || boards >= budget) {
      diagnostics.push({
        offset: pos,
        message:
          result.error || `每篇正文最多渲染 ${MAX_SYNTAX_BOARDS} 个语法棋盘。`,
      });
      pending += raw;
    } else {
      flush();
      boards++;
      parts.push({ ...result.board, source: raw });
    }
    pos = close + 2;
  }
  flush();
  return { parts, diagnostics, boards };
}

export function renderDocumentBlocks(body) {
  let remaining = MAX_SYNTAX_BOARDS;
  return (body?.blocks || []).flatMap((block) => {
    if (block.type !== "paragraph") return [block];
    const parsed = parseBoardText(block.text, remaining);
    remaining -= parsed.boards;
    return parsed.parts;
  });
}
