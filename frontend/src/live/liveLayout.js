import { onMounted, onUnmounted } from 'vue';

export const livePageScale = (width, height = Infinity) => Math.min(.85, Math.max(1, width) / 1500, Math.max(1, height) / 1080);

// Keep the existing layout at ordinary ratios; letterbox only extremely wide windows.
export function livePageWidth(width, height) {
  const scale = livePageScale(width, height);
  return Math.max(1500, Math.min(width, Math.max(1, height) * 2) / scale);
}

export function useLiveLayoutScale() {
  const update = () => {
    // The previous layout may briefly have a scrollbar during a resize. Its
    // clientWidth must not determine the next scale and leave a stale gutter.
    const width = window.innerWidth, height = window.innerHeight;
    document.body.style.setProperty('--live-scale', String(livePageScale(width, height)));
    document.body.style.setProperty('--live-page-width', `${livePageWidth(width, height)}px`);
  };
  update();
  onMounted(() => {
    window.addEventListener('resize', update);
    window.addEventListener('orientationchange', update);
  });
  onUnmounted(() => {
    window.removeEventListener('resize', update);
    window.removeEventListener('orientationchange', update);
    document.body.style.removeProperty('--live-scale');
    document.body.style.removeProperty('--live-page-width');
  });
}

// DOM rectangles use the shared host viewport utility.
export { layoutViewport as liveLayoutViewport } from '../utils/layoutViewport.js';
