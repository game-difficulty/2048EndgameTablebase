export function refocusBoardHotkeyTarget(targetRef, sourceElement = null) {
  if (sourceElement instanceof HTMLElement) {
    sourceElement.blur();
  }

  requestAnimationFrame(() => {
    const target = targetRef?.value;
    if (target instanceof HTMLElement) {
      target.focus({ preventScroll: true });
    }
  });
}
