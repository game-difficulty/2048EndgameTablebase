// The whole board owns the gesture, including padding, gaps and empty cells.
// One contact produces one move as soon as its direction is clear.
export function createBoardSwipe(disabled, move) {
  let contact = null;
  function cancel(event) {
    if (event && contact?.id !== event.pointerId) return;
    const previous = contact;
    contact = null;
    try {
      if (previous?.target.hasPointerCapture?.(previous.id)) previous.target.releasePointerCapture(previous.id);
    } catch { /* The browser may already have released capture. */ }
  }
  function down(event) {
    if (disabled() || event.isPrimary === false || contact || (event.pointerType === 'mouse' && event.button !== 0)) return;
    event.preventDefault();
    event.stopPropagation(); // Nested cargo/number boards must not both move.
    contact = { id: event.pointerId, x: event.clientX, y: event.clientY, target: event.currentTarget, fired: false };
    try { event.currentTarget.setPointerCapture?.(event.pointerId); } catch { /* Best effort. */ }
  }
  function drag(event) {
    if (!contact || contact.id !== event.pointerId) return;
    if (disabled()) { cancel(); return; }
    event.preventDefault();
    if (contact.fired) return;
    const dx = event.clientX - contact.x, dy = event.clientY - contact.y;
    if (Math.max(Math.abs(dx), Math.abs(dy)) < 18) return;
    contact.fired = true;
    move(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'right' : 'left') : (dy > 0 ? 'down' : 'up'));
  }
  function up(event) {
    drag(event); // Fallback for browsers that coalesce the last pointermove.
    cancel(event);
  }
  return { down, drag, up, cancel };
}
