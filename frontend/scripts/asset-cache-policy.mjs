const filenames = {
  entryFileNames: 'assets/[name]-[hash].js',
  chunkFileNames: 'assets/[name]-[hash].js',
  assetFileNames: 'assets/[name]-[hash][extname]',
};

export function assertStableAssetNames(output = {}) {
  for (const [key, expected] of Object.entries(filenames)) {
    const actual = output[key];
    if (actual === undefined || actual === expected) continue;
    // The maintained chunk function only coalesces shared module names.
    if (key === 'chunkFileNames' && typeof actual === 'function') continue;
    throw new Error(`${key} must use ${expected}; keep release IDs in HTML/release.json, not in every asset URL`);
  }
}

export function assetCachePolicy() {
  return {
    name: 'stable-asset-cache-policy',
    apply: 'build',
    configResolved(config) {
      const output = config.build.rollupOptions?.output;
      for (const item of Array.isArray(output) ? output : [output]) assertStableAssetNames(item);
    },
  };
}
