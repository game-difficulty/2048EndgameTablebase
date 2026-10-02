import MarkdownIt from "markdown-it";
export const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  typographer: false,
});
markdown.inline.ruler.before("autolink", "mention", (state, silent) => {
  if (state.linkLevel) return false;
  const match = /^<@([1-9][0-9]{0,14})>/.exec(state.src.slice(state.pos));
  if (!match) return false;
  if (!silent) {
    const token = state.push("mention", "", 0);
    token.content = match[1];
  }
  state.pos += match[0].length;
  return true;
});
const originalValidate = markdown.validateLink;
markdown.validateLink = (url) =>
  originalValidate(url) &&
  (/^https?:\/\//i.test(url) ||
    /^mailto:/i.test(url) ||
    /^\/(?!\/)/.test(url) ||
    /^#/.test(url));
export function mediaId(url) {
  return (
    /^\/api\/forum\/v1\/media\/([0-9a-f-]{36})$/i.exec(url || "")?.[1] || null
  );
}
