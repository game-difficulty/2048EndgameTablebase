import { inject, onMounted, onBeforeUnmount, watch } from 'vue';

export const roomSurfaceHost = Symbol('room-surface-host');

// Layout pixels, not screen pixels: the room may be zoomed and its canvas
// transformed. Reading boundingClientRect here would apply that scale twice.
export function observeSurfaceBox(element, update) {
  const host = element.ownerDocument.defaultView;
  let frame = null;
  const measure = () => {
    const style = host.getComputedStyle(element);
    update(parseFloat(style.width) || 0, parseFloat(style.height) || 0);
  };
  const observer = host.ResizeObserver && new host.ResizeObserver(entries => {
    const { width, height } = entries[0].contentRect;
    update(width, height);
  });
  // Room zoom/width are also updated by resize listeners. Read after all those
  // listeners, including on engines where no ResizeObserver follows up.
  const resized = () => {
    if (frame !== null) return;
    frame = host.requestAnimationFrame(() => { frame = null; measure(); });
  };
  observer?.observe(element);
  host.addEventListener('resize', resized);
  measure();
  return () => {
    observer?.disconnect();
    host.removeEventListener('resize', resized);
    if (frame !== null) host.cancelAnimationFrame(frame);
  };
}

// Rebind descendants when their existing DOM is adopted into/out of PiP.
export function useSurfaceBox(element, update) {
  const host = inject(roomSurfaceHost, null);
  let stop;
  const refresh = () => {
    stop?.();
    if (element.value) stop = observeSurfaceBox(element.value, update);
  };
  onMounted(refresh);
  if (host) watch(host, refresh, { flush: 'post' });
  onBeforeUnmount(() => stop?.());
  return refresh;
}
