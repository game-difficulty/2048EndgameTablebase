// Each effective spawn consumes exactly one ticket. The ticket's independent
// channels decide position, value, and special-spawn timing without advancing
// the stream again. Rule-specific streams never consume spawn tickets.
export function seed32(value) {
  let hash = 2166136261;
  for (const char of String(value)) {
    hash ^= char.codePointAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0 || 0x9e3779b9;
}

export function nextRandom(state) {
  let value = state >>> 0;
  value ^= value << 13;
  value ^= value >>> 17;
  value ^= value << 5;
  return value >>> 0;
}

export function randomStream(seed, domain) {
  let state = seed32(`${seed}:${domain}`);
  return () => {
    state = nextRandom(state);
    return state / 0x100000000;
  };
}

export function ticketFloat(ticket, channel) {
  const salt = {
    position: 0xa511e9b3,
    value: 0x63d83595,
    special: 0xc2b2ae35,
  }[channel];
  if (salt == null) throw new Error(`Unknown spawn ticket channel: ${channel}`);
  // Avalanche before deriving channels: xorshift(ticket ^ salt) alone keeps
  // different channels linearly correlated, biasing special-spawn positions.
  let value = (ticket ^ salt) >>> 0;
  value ^= value >>> 16;
  value = Math.imul(value, 0x85ebca6b);
  value ^= value >>> 13;
  value = Math.imul(value, 0xc2b2ae35);
  value ^= value >>> 16;
  return (value >>> 0) / 0x100000000;
}
