import { readFileSync } from 'node:fs';

export function startupRecovery({ diagnostics = true } = {}) {
  const source = readFileSync(new URL('../public/startup-recovery.js', import.meta.url), 'utf8');
  const panel = `<style>
#app-startup{min-height:100vh;box-sizing:border-box;padding:48px 20px;text-align:center;font:16px/1.5 Arial,sans-serif;color:#475569;background:#f8fafc;background:var(--bg-main,#f8fafc)}
html[data-theme="dark"] #app-startup{color:#d7e0eb;background:#0f172a;background:var(--bg-main,#0f172a)}
#app-startup button{margin-top:16px;padding:8px 16px;border:1px solid currentColor;border-radius:6px;background:transparent;color:inherit;font:inherit;cursor:pointer}
</style><div id="app-startup" role="status" aria-live="polite"><span data-startup-message>Loading...</span><br><button type="button" style="display:none">Reload page</button></div>`;
  return {
    name: 'startup-recovery',
    transformIndexHtml: {
      order: 'pre',
      handler(html) {
        const script = `<script>${source.replace(/<\/script/gi, '<\\/script')}</script>`;
        const meta = diagnostics ? '<meta name="load-diagnostics" content="/api/client-diagnostics">' : '';
        return html.replace('<head>', `<head>${meta}${script}`)
          .replace('<body>', `<body>${panel}<script>window.__APP_RECOVERY__.render();</script>`);
      },
    },
  };
}
