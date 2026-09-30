export const CHAT_LIMIT = 80;

// Keep the ranges and joined-emoji handling identical to backend/live/chat_length.py.
export function chatLength(text) {
  let total = 0, joined = false, regional = false;
  for (const char of text) {
    const n = char.codePointAt(0);
    if (n === 0x200D) { joined = true; continue; }
    if ((n >= 0x300 && n <= 0x36F) || (n >= 0xFE00 && n <= 0xFE0F)
      || (n >= 0x1F3FB && n <= 0x1F3FF) || (n >= 0xE0100 && n <= 0xE01EF)
      || n === 0x20E3) continue;
    const flag = n >= 0x1F1E6 && n <= 0x1F1FF;
    if (joined || (flag && regional)) { joined = false; regional = false; continue; }
    regional = flag;
    const wide = (n >= 0x1100 && n <= 0x11FF) || (n >= 0x2600 && n <= 0x27BF)
      || (n >= 0x2E80 && n <= 0xA4CF) || (n >= 0xAC00 && n <= 0xD7AF)
      || (n >= 0xF900 && n <= 0xFAFF) || (n >= 0xFE10 && n <= 0xFE6F)
      || (n >= 0xFF01 && n <= 0xFF60) || n >= 0x1F000;
    total += wide ? 2 : 1;
  }
  return total;
}
