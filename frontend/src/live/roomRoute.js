export function roomIdFromPath(path) {
  if (['/', '/live', '/live/', '/live/index.html'].includes(path)) return 'ai-classic';
  return path.match(/^\/(?:live\/)?rooms\/([a-z0-9][a-z0-9-]{0,47})\/?$/)?.[1] || null;
}
