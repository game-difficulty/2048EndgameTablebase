let modulePromise = null;
let generator = null;

const runtimeImport = url => Function('specifier', 'return import(specifier)')(url);

function encodeBoard(board) {
  return board.slice(0, 16).reduce((encoded, value, index) => {
    const exponent = value > 0 ? Math.min(15, Math.log2(value) | 0) : 0;
    return encoded | (BigInt(exponent) << BigInt((15 - index) * 4));
  }, 0n);
}

export async function evilSpawnRuntime(board, depth, tieSeed) {
  modulePromise ||= runtimeImport('/wasm/evil_core.js?v=tournament-v2').then(({ default: create }) => create({
    locateFile: path => `/wasm/${path}?v=tournament-v2`,
  }));
  const module = await modulePromise;
  const encoded = encodeBoard(board);
  generator ||= new module.EvilGen(encoded);
  generator.reset_board(encoded);
  const result = typeof generator.gen_new_num_seeded === 'function'
    ? generator.gen_new_num_seeded(depth, tieSeed >>> 0)
    : generator.gen_new_num(depth);
  const index = Number(result?.[1]);
  const exponent = Number(result?.[2]);
  if (!Number.isInteger(index) || board[index] !== 0 || ![1, 2].includes(exponent)) {
    throw new Error('EvilGen returned an invalid spawn.');
  }
  return { index, value: 2 ** exponent, depth };
}
