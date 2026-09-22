import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'
import { buildRenderCompat } from './scripts/build-render-compat.mjs'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), tailwindcss(), {
    name: 'build-render-compat',
    apply: 'build',
    async closeBundle() { await buildRenderCompat(); },
  }],
  build: { manifest: true, rollupOptions: { input: { main: 'index.html', live: 'live/index.html' } } },
})
