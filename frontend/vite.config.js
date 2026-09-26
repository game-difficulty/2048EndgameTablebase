import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'
import { buildRenderCompat } from './scripts/build-render-compat.mjs'
import { precompress } from './scripts/precompress.mjs'
import { buildRelease } from './scripts/build-release.mjs'

function liveRoomEntry(req, res, next) {
  if (/^\/(?:live\/)?rooms\//.test(req.url || '') || /^\/lobby\/?(?:\?.*)?$/.test(req.url || '')) req.url = '/live/index.html';
  if (/^\/user\/[^/?#]+\/?(?:\?.*)?$/.test(req.url || '')) req.url = '/human/index.html';
  if (/^\/leaderboard\/?(?:\?.*)?$/.test(req.url || '')) req.url = '/human/index.html';
  next();
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), tailwindcss(), buildRelease(), {
    name: 'live-room-entry',
    configureServer(server) { server.middlewares.use(liveRoomEntry); },
    configurePreviewServer(server) { server.middlewares.use(liveRoomEntry); },
  }, {
    name: 'build-render-compat',
    apply: 'build',
    async closeBundle() { await buildRenderCompat(); await precompress(); },
  }],
  build: { manifest: true, rollupOptions: { input: { main: 'index.html', live: 'live/index.html', human: 'human/index.html' } } },
})
