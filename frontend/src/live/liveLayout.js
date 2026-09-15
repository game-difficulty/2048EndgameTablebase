import { onMounted, onUnmounted } from 'vue';

export const livePageScale = (width, height = Infinity) => Math.min(.85, Math.max(1, width) / 1500, Math.max(1, height) / 1080);

export function useLiveLayoutScale() {
  let observer;
  const update = () => document.body.style.setProperty('--live-scale', String(livePageScale(document.documentElement.clientWidth, window.innerHeight)));
  update();
  onMounted(() => {
    observer = new ResizeObserver(update);
    observer.observe(document.documentElement);
    window.addEventListener('resize', update);
  });
  onUnmounted(() => {
    observer?.disconnect();
    window.removeEventListener('resize', update);
    document.body.style.removeProperty('--live-scale');
  });
}

export function useLiveStageFit(stageRef) {
  let observer, frame;
  const update = () => {
    frame = null;
    const stage = stageRef.value;
    if (!stage) return;
    const grid = stage.querySelector('.broadcast-grid');
    const column = stage.querySelector('.board-column');
    const board = stage.querySelector('.board-stage');
    if (!grid || !column || !board) return;
    const scale = Number.parseFloat(getComputedStyle(document.body).getPropertyValue('--live-scale')) || 1;
    const rect = stage.getBoundingClientRect();
    // Reserve the actual title, gifts, scores and code heights before sizing the board.
    const gridHeight = grid.getBoundingClientRect().height;
    const outsideGrid = rect.height - gridHeight;
    const outsideBoard = column.getBoundingClientRect().height - board.getBoundingClientRect().height;
    const available = Math.max(0, (window.innerHeight - rect.top - 12 - outsideGrid) / scale);
    stage.style.setProperty('--live-grid-limit', `${Math.floor(available)}px`);
    stage.style.setProperty('--live-board-limit', `${Math.max(160, Math.floor(available - outsideBoard / scale))}px`);
  };
  const schedule = () => { if (frame == null) frame = requestAnimationFrame(update); };
  onMounted(() => {
    observer = new ResizeObserver(schedule);
    const stage = stageRef.value;
    for (const element of [stage, ...stage.querySelectorAll('.live-title, .broadcast-grid, .board-column, .board-stage, .gift-panel')]) observer.observe(element);
    window.addEventListener('resize', schedule);
    schedule();
  });
  onUnmounted(() => {
    observer?.disconnect();
    cancelAnimationFrame(frame);
    window.removeEventListener('resize', schedule);
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
