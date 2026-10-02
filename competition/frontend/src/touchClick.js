// Activate a touch tap directly instead of relying on a synthetic mouse click.
// Cancelling touchend suppresses that synthetic click, so undo runs only once.
export function bindTouchClick(element) {
  let contact = null;
  const cancel = () => { contact = null; };
  const start = event => {
    contact = null;
    if (element.disabled || event.touches.length !== 1) return;
    const touch = event.touches[0];
    contact = { id: touch.identifier, x: touch.clientX, y: touch.clientY };
  };
  const move = event => {
    const touch = Array.from(event.touches).find(item => item.identifier === contact?.id);
    if (!touch || event.touches.length !== 1
      || Math.hypot(touch.clientX - contact.x, touch.clientY - contact.y) > 10) cancel();
  };
  const end = event => {
    const previous = contact;
    cancel();
    if (!previous || element.disabled) return;
    const touch = Array.from(event.changedTouches).find(item => item.identifier === previous.id);
    if (!touch || Math.hypot(touch.clientX - previous.x, touch.clientY - previous.y) > 10) return;
    const rect = element.getBoundingClientRect();
    if (touch.clientX < rect.left || touch.clientX > rect.right
      || touch.clientY < rect.top || touch.clientY > rect.bottom) return;
    event.preventDefault();
    element.click();
  };
  const handlers = { touchstart: start, touchmove: move, touchend: end, touchcancel: cancel };
  for (const [name, handler] of Object.entries(handlers)) {
    element.addEventListener(name, handler, { passive: name !== 'touchend' });
  }
  return () => {
    for (const [name, handler] of Object.entries(handlers)) element.removeEventListener(name, handler);
  };
}

const cleanups = new WeakMap();
export const vTouchClick = {
  mounted(element) { cleanups.set(element, bindTouchClick(element)); },
  unmounted(element) { cleanups.get(element)?.(); cleanups.delete(element); },
};
