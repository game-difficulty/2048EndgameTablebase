let aiCorePromise = null;
let evilCorePromise = null;

const runtimeImport = (url) => Function('specifier', 'return import(specifier)')(url);
const AI_CORE_VERSION = 'prune-relaxation-20260917';
const EVIL_CORE_VERSION = 'evil-20260706';

export async function getAiCore() {
  if (!aiCorePromise) {
    aiCorePromise = runtimeImport(`/wasm/ai_core.js?v=${AI_CORE_VERSION}`).then(({ default: createAICore }) =>
      createAICore({
        locateFile: (path) => `/wasm/${path}?v=${AI_CORE_VERSION}`,
      })
    );
  }
  return aiCorePromise;
}

export async function getEvilCore() {
  if (!evilCorePromise) {
    evilCorePromise = runtimeImport(`/wasm/evil_core.js?v=${EVIL_CORE_VERSION}`).then(({ default: createEvilCore }) =>
      createEvilCore({
        locateFile: (path) => `/wasm/${path}?v=${EVIL_CORE_VERSION}`,
      })
    );
  }
  return evilCorePromise;
}
