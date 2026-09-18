import { onMounted, onUnmounted } from 'vue';

export const livePageScale = (width, height = Infinity) => Math.min(.85, Math.max(1, width) / 1500, Math.max(1, height) / 1080);

// Keep the existing layout at ordinary ratios; letterbox only extremely wide windows.
export function livePageWidth(width, height) {
  const scale = livePageScale(width, height);
  return Math.max(1500, Math.min(width, Math.max(1, height) * 2) / scale);
}

export function useLiveLayoutScale() {
  const update = () => {
    const width = document.documentElement.clientWidth, height = window.innerHeight;
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

// DOM rectangles are visual pixels; fixed-position offsets use zoomed CSS pixels.
export function liveLayoutViewport(element) {
  const scale = Number.parseFloat(getComputedStyle(document.body).getPropertyValue('--live-scale')) || 1;
  const rect = element.getBoundingClientRect();
  return {
    width: innerWidth / scale,
    height: innerHeight / scale,
    rect: Object.fromEntries(['left', 'top', 'right', 'bottom', 'width', 'height'].map(key => [key, rect[key] / scale])),
  };
}
