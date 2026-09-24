import { execFileSync } from 'node:child_process';

export function buildRelease() {
  const revision = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  const builtAt = new Date().toISOString();
  const buildId = `${revision.slice(0, 12)}-${builtAt.replace(/[^0-9]/g, '')}`;
  return {
    name: 'build-release',
    apply: 'build',
    transformIndexHtml() {
      return [{ tag: 'meta', attrs: { name: 'build-id', content: buildId }, injectTo: 'head' }];
    },
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'release.json',
        source: JSON.stringify({ buildId, revision, builtAt }) + '\n' });
    },
  };
}
