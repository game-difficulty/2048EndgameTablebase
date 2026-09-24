// DOM rectangles are visual pixels; fixed-position offsets use zoomed CSS pixels.
export function layoutViewport(element) {
  const scale = Number.parseFloat(getComputedStyle(document.body).zoom) || 1;
  const rect = element.getBoundingClientRect();
  return {
    width: innerWidth / scale,
    height: innerHeight / scale,
    rect: Object.fromEntries(['left', 'top', 'right', 'bottom', 'width', 'height'].map(key => [key, rect[key] / scale])),
  };
}
