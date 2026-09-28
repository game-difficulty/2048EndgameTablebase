import { evilSpawnRuntime } from './evilSpawnRuntime.js';

self.addEventListener('message', async ({ data }) => {
  try {
    const result = await evilSpawnRuntime(data.board, data.depth, data.tieSeed);
    self.postMessage({ id: data.id, result });
  } catch (error) {
    self.postMessage({
      id: data.id,
      error: error instanceof Error ? error.message : String(error),
    });
  }
});
