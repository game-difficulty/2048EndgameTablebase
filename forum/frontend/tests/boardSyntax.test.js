import test from "node:test";
import assert from "node:assert/strict";
import {
  decodeBoardSyntax,
  encodeBoardSyntax,
  parseBoardText,
  renderDocumentBlocks,
} from "../src/boardSyntax.js";
import {
  parsePracticeHex,
  practiceBoardHex,
} from "../../../frontend/src/human/practice.js";

test("matches Play row-major encoding, including high tiles and shortened codes", () => {
  for (const [variant, count] of [
    ["4x4", 16],
    ["3x4", 12],
    ["3x3", 9],
    ["2x4", 8],
  ]) {
    for (const code of ["12", "0xAbC", "0", "123fgh".padEnd(count, "0")]) {
      const { board } = decodeBoardSyntax(`[[board:${variant}:${code}]]`);
      assert.deepEqual(board.cells, parsePracticeHex(code, count));
      assert.equal(
        encodeBoardSyntax(board),
        `[[board:${variant}:${practiceBoardHex(board.cells)}]]`,
      );
    }
  }
  assert.deepEqual(
    decodeBoardSyntax("[[board:2x4:12345678]]").board.cells,
    [2, 4, 8, 16, 32, 64, 128, 256],
  );
});

test("row separators, case, multiplication sign, optional caption and extended alphabet", () => {
  const { board } = decodeBoardSyntax(
    "[[BOARD:2×4:0xGHIJ/K000|讨论 | <img src=x onerror=alert(1)>]]",
  );
  assert.deepEqual(
    board.cells,
    [65536, 131072, 262144, 524288, 1048576, 0, 0, 0],
  );
  assert.equal(board.caption, "讨论 | <img src=x onerror=alert(1)>");
  assert.deepEqual(decodeBoardSyntax(encodeBoardSyntax(board)).board, board);
});

for (const input of [
  "[[board:5x5:123]]",
  "[[board:__proto__:123]]",
  "[[board:4x4:]]",
  "[[board:2x4:123456789]]",
  "[[board:2x4:123/45678]]",
  "[[board:3x4:1234/5678]]",
  "[[board:4x4:zzzz]]",
  "[[board:4x4:1 2]]",
  "[[board:4x4:-1]]",
  "[[board:4x4:1\n2]]",
  "[[board:4x4:1|" + "长".repeat(501) + "]]",
])
  test(`invalid syntax stays literal: ${input.slice(0, 45)}`, () => {
    const result = parseBoardText(input);
    assert.equal(result.boards, 0);
    assert.deepEqual(result.parts, [{ type: "paragraph", text: input }]);
    assert.equal(result.diagnostics.length, 1);
  });

test("interleaves discussion with multiple boards without losing text", () => {
  const result = parseBoardText(
    "之前\n[[board:4x4:1|第一张]]中间[[board:3x3:2]]\n之后",
  );
  assert.deepEqual(
    result.parts.map((p) => p.type),
    ["paragraph", "board", "paragraph", "board", "paragraph"],
  );
  assert.equal(result.parts[0].text, "之前\n");
  assert.equal(result.parts[2].text, "中间");
  assert.equal(result.parts[4].text, "\n之后");
  assert.equal(result.parts[1].source, "[[board:4x4:1|第一张]]");
});

test("escaped and backtick examples are literal, including unclosed code", () => {
  const syntax = "[[board:4x4:1]]";
  for (const source of [
    "`" + syntax + "`",
    "```text\n" + syntax + "\n```",
    "`` ` " + syntax + " ``",
    "`" + syntax,
  ]) {
    assert.deepEqual(parseBoardText(source).parts, [
      { type: "paragraph", text: source },
    ]);
  }
  assert.deepEqual(parseBoardText("\\" + syntax).parts, [
    { type: "paragraph", text: syntax },
  ]);
  assert.equal(parseBoardText("`" + syntax + "` " + syntax).boards, 1);
});

test("incomplete token does not swallow a later complete token", () => {
  const parsed = parseBoardText("[[board:4x4:12\n[[board:3x3:123]]");
  assert.equal(parsed.boards, 1);
  assert.equal(parsed.diagnostics.length, 1);
  assert.equal(parsed.parts[0].text, "[[board:4x4:12\n");
  assert.equal(parseBoardText("[[board:4x4:12").diagnostics.length, 1);
});

test("render budget applies across paragraphs and keeps excess source", () => {
  const text = "[[board:4x4:1]]";
  const blocks = renderDocumentBlocks({
    blocks: Array.from({ length: 40 }, () => ({ type: "paragraph", text })),
  });
  assert.equal(blocks.filter((b) => b.type === "board").length, 20);
  assert.equal(blocks.filter((b) => b.type === "paragraph").length, 20);
  assert.equal(parseBoardText(text.repeat(21)).diagnostics.length, 1);
});

test("drawing export is lossless or explicitly unavailable", () => {
  const board = {
    rows: 2,
    cols: 4,
    cells: [0, 2, 4, 8, 16, 32768, 65536, 1048576],
    caption: "解释",
  };
  assert.deepEqual(decodeBoardSyntax(encodeBoardSyntax(board)).board, {
    type: "board",
    ...board,
  });
  assert.equal(
    encodeBoardSyntax({ ...board, cells: [3, ...board.cells.slice(1)] }),
    null,
  );
  assert.equal(encodeBoardSyntax({ ...board, caption: "说明]]后文" }), null);
});

test("old structured boards and unknown document blocks are preserved", () => {
  const board = { type: "board", rows: 2, cols: 4, cells: Array(8).fill(0) };
  assert.deepEqual(
    renderDocumentBlocks({ blocks: [board, { type: "future" }] }),
    [board, { type: "future" }],
  );
});
