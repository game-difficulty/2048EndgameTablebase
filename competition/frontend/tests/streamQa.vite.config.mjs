import { mergeConfig } from 'vite';
import base from '../../../frontend/vite.config.js';
import { fileURLToPath } from 'node:url';
export default mergeConfig(base, { server: { fs: { allow: [fileURLToPath(new URL('../../../', import.meta.url))] } } });
