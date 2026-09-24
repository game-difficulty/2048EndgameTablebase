import { readdir, readFile, writeFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';

export async function precompress(directory = new URL('../dist/', import.meta.url)) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = new URL(entry.name + (entry.isDirectory() ? '/' : ''), directory);
    if (entry.isDirectory()) await precompress(file);
    else if (/\.(html|js|css|json|svg|wasm|xml|txt)$/.test(entry.name)) {
      const data = await readFile(file);
      const compressed = gzipSync(data, { level: 9 });
      if (data.length >= 512 && compressed.length < data.length) await writeFile(new URL(`${entry.name}.gz`, directory), compressed);
    }
  }
}
