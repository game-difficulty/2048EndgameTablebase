import { readFile } from 'node:fs/promises';
import { resolveTileColors } from '../src/utils/tileColors.js';

export function replayThemeAssets() {
  async function source() {
    const themes = JSON.parse(await readFile(new URL('../../docs_and_configs/themes.json', import.meta.url), 'utf8'));
    const palettes = Object.fromEntries(Object.entries(themes).map(([name, colors]) => [name, resolveTileColors(colors)]));
    return `window.ReplayThemeCatalog=${JSON.stringify(palettes)};`;
  }
  return {
    name: 'replay-theme-assets',
    async generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'verse-replay/theme-catalog.js', source: await source() });
    },
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        if (/^\/verse-replay\/(?:\?|$)/.test(req.url || '')) {
          req.url = req.url.replace('/verse-replay/', '/verse-replay/index.html');
          return next();
        }
        if ((req.url || '').split('?')[0] !== '/verse-replay/theme-catalog.js') return next();
        try {
          res.setHeader('Content-Type', 'text/javascript; charset=utf-8');
          res.end(await source());
        } catch (error) { next(error); }
      });
    },
  };
}
