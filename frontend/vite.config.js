import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'
import { startupRecovery } from './scripts/startup-recovery.mjs'
import { assetCachePolicy } from './scripts/asset-cache-policy.mjs'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue(), tailwindcss(), startupRecovery({ diagnostics: false }), assetCachePolicy()],
})
