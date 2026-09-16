// Only emphasis is supported; all content remains escaped Vue text, never HTML.
export function emphasisParts(text) {
  return String(text).split(/(\*\*[^*]+\*\*)/g).filter(Boolean).map(part => ({
    bold: part.startsWith('**') && part.endsWith('**'),
    text: part.startsWith('**') && part.endsWith('**') ? part.slice(2, -2) : part,
  }));
}
