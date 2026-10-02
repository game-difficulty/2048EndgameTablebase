import test from "node:test";
import assert from "node:assert/strict";
import { markdown, mediaId } from "../src/markdown.js";
import { parseBoardText } from "../src/boardSyntax.js";

test("mentions skip escaped examples, code, image alt text and existing links", () => {
  const tokens = markdown
    .parse(
      "<@2> **<@3>** `<@4>` \\<@5> [<@6>](https://example.org) ![<@7>](/image)",
      {},
    )
    .flatMap((t) => t.children || []);
  assert.deepEqual(
    tokens.filter((t) => t.type === "mention").map((t) => t.content),
    ["2", "3"],
  );
  assert.ok(
    !markdown
      .parse("```\n<@2>\n```\n\n    <@3>", {})
      .some((t) => t.children?.some((c) => c.type === "mention")),
  );
});

test("HTML and unsafe URL schemes never become active markdown tokens", () => {
  const text =
    "<script>alert(1)</script> [x](javascript:alert(1)) ![x](data:text/html,x)";
  const tokens = markdown.parse(text, {}).flatMap((t) => t.children || [t]);
  assert.ok(
    !tokens.some(
      (t) =>
        t.type === "html_inline" ||
        t.type === "html_block" ||
        t.type === "link_open" ||
        t.type === "image",
    ),
  );
  assert.equal(mediaId("https://external.invalid/tracker.png"), null);
});
test("rich text keeps structure and local attachments have explicit identifiers", () => {
  const tokens = markdown.parse(
    "## 标题\n\n**重点**\n\n> 引用\n\n- 列表\n\n```js\n1<2\n```",
    {},
  );
  assert.ok(tokens.some((t) => t.type === "heading_open"));
  assert.ok(tokens.some((t) => t.type === "blockquote_open"));
  assert.ok(tokens.some((t) => t.type === "bullet_list_open"));
  assert.ok(tokens.some((t) => t.type === "fence"));
  assert.equal(
    mediaId("/api/forum/v1/media/12345678-1234-1234-1234-123456789abc"),
    "12345678-1234-1234-1234-123456789abc",
  );
});
test("replay anchors render while escaped and code examples stay literal", () => {
  const source = "[[replay:12345678-1234-1234-1234-123456789abc@42]]";
  const parsed = parseBoardText(source);
  assert.equal(parsed.parts[0].type, "replay");
  assert.equal(parsed.parts[0].step, 42);
  assert.equal(parseBoardText("`" + source + "`").boards, 0);
  assert.equal(parseBoardText("\\" + source).boards, 0);
});
