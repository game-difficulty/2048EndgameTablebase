import { h } from "vue";
import { RouterLink } from "vue-router";
import { markdown, mediaId } from "../markdown";
import AttachmentImage from "./AttachmentImage.vue";

// Render parser tokens as Vue nodes: user text is never inserted as HTML.
function nodes(tokens) {
  const root = [],
    stack = [root];
  for (const t of tokens) {
    let current = stack.at(-1);
    if (t.hidden) {
      if (t.nesting === 1) stack.push(current);
      else if (t.nesting === -1) stack.pop();
      continue;
    }
    if (t.type === "inline") {
      current.push(...nodes(t.children || []));
      continue;
    }
    if (t.type === "mention") {
      current.push(
        h(
          RouterLink,
          { to: "/u/" + t.content, class: "mention" },
          () => "@" + t.content,
        ),
      );
      continue;
    }
    if (t.type === "text" || t.type === "code_inline") {
      current.push(t.type === "text" ? t.content : h("code", t.content));
      continue;
    }
    if (t.type === "fence" || t.type === "code_block") {
      if (t.type === "fence" && /^details(?:\s|$)/.test(t.info.trim())) {
        current.push(
          h("details", { class: "collapsed-text" }, [
            h("summary", t.info.trim().slice(7).trim() || "展开内容"),
            h("p", { style: "white-space:pre-wrap" }, t.content),
          ]),
        );
      } else current.push(h("pre", [h("code", t.content)]));
      continue;
    }
    if (t.type === "softbreak" || t.type === "hardbreak") {
      current.push(h("br"));
      continue;
    }
    if (t.type === "image") {
      const id = mediaId(t.attrGet("src"));
      current.push(
        id
          ? h(AttachmentImage, { id, alt: t.content })
          : h(
              "span",
              { class: "muted" },
              `[图片：${t.content || "请上传图片到论坛"}]`,
            ),
      );
      continue;
    }
    if (t.nesting === -1) {
      stack.pop();
      continue;
    }
    const allowed = new Set([
      "p",
      "strong",
      "em",
      "s",
      "blockquote",
      "ul",
      "ol",
      "li",
      "h1",
      "h2",
      "h3",
      "h4",
      "h5",
      "h6",
      "hr",
      "table",
      "thead",
      "tbody",
      "tr",
      "th",
      "td",
      "a",
    ]);
    if (!allowed.has(t.tag)) {
      current.push(t.content || "");
      continue;
    }
    const attrs = {};
    if (t.tag === "a") {
      attrs.href = t.attrGet("href");
      attrs.title = t.attrGet("title");
      attrs.rel = "nofollow noopener noreferrer";
      if (/^https?:/.test(attrs.href || "")) attrs.target = "_blank";
    }
    if (t.tag === "ol" && t.attrGet("start")) attrs.start = t.attrGet("start");
    const children = [];
    current.push(h(t.tag, attrs, children));
    if (t.nesting === 1) stack.push(children);
  }
  return root;
}
export default {
  props: { text: String },
  setup: (props) => () =>
    h(
      "div",
      { class: "rich-content" },
      nodes(markdown.parse(props.text || "", {})),
    ),
};
