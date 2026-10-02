import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { fileURLToPath } from "node:url";

export default defineConfig({
  plugins: [vue()],
  server: {
    fs: {
      allow: [
        fileURLToPath(new URL(".", import.meta.url)),
        fileURLToPath(new URL("../../font", import.meta.url)),
        fileURLToPath(new URL("../../frontend/src/utils", import.meta.url)),
        fileURLToPath(
          new URL("../../docs_and_configs/themes.json", import.meta.url),
        ),
      ],
    },
    proxy: {
      "/api": "http://127.0.0.1:8002",
      "/health": "http://127.0.0.1:8002",
    },
  },
  build: { manifest: true },
});
