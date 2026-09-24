export function decodeReceipt(value) {
  if (!Array.isArray(value)) return value;
  if (value.length !== 6 || value[0] !== 2 || !Number.isInteger(value[3]) || value[3] < 0 || value[3] > 14) throw new Error('invalid_receipt');
  const [, seq, epoch, flags, milliseconds, permit] = value;
  const status = ['active', 'pending_archive', 'sealed'][flags & 3];
  if (!status || !Number.isInteger(seq) || !Number.isInteger(epoch) || !Number.isFinite(milliseconds) || typeof permit !== 'string') throw new Error('invalid_receipt');
  return { seq, epoch, status, monitored: !!(flags & 4), eligibility: flags & 8 ? 'disqualified' : 'eligible',
    server_time: milliseconds / 1000, permit_until: permit ? Number(permit.split('.')[0]) / 1000 : 0, permit };
}

export async function uploadBody(raw) {
  const plain = { body: raw, headers: {} };
  if (raw.byteLength < 4096 || typeof CompressionStream !== 'function') return plain;
  const count = raw.byteLength / 5, planes = new Uint8Array(raw.byteLength);
  for (let i = 0; i < count; i++) for (let lane = 0; lane < 5; lane++) planes[lane * count + i] = raw[i * 5 + lane];
  try {
    const stream = new Blob([planes]).stream().pipeThrough(new CompressionStream('gzip'));
    const compressed = new Uint8Array(await new Response(stream).arrayBuffer());
    // Include new header overhead; tiny/small wins are not worth changing format.
    if (compressed.byteLength + 64 < raw.byteLength) return {
      body: compressed, headers: { 'Content-Encoding': 'gzip', 'X-Human-Layout': 'planes5' },
    };
  } catch { /* Older browsers retain the exact same canonical upload. */ }
  return plain;
}
