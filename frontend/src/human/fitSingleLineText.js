export function fittedFontSize(baseSize, availableWidth, requiredWidth, minimumRatio = 0.5) {
  const base = Number(baseSize) || 0;
  const available = Number(availableWidth) || 0;
  const required = Number(requiredWidth) || 0;
  if (base <= 0 || available <= 0 || required <= available) return base;
  return Math.max(base * minimumRatio, base * available / required * 0.98);
}

const observers = new WeakMap();

function fit(element) {
  element.style.fontSize = '';
  const base = Number.parseFloat(getComputedStyle(element).fontSize) || 13;
  const size = fittedFontSize(base, element.clientWidth, element.scrollWidth);
  if (size < base) element.style.fontSize = `${size}px`;
}

function schedule(element) {
  cancelAnimationFrame(element.__rankingFitFrame || 0);
  element.__rankingFitFrame = requestAnimationFrame(() => fit(element));
}

function scheduleWhenTextChanges(element) {
  const text = element.textContent;
  if (element.__rankingFitText === text) return;
  element.__rankingFitText = text;
  schedule(element);
}

export const fitSingleLineText = {
  mounted(element) {
    let cleanup;
    if (typeof ResizeObserver === 'function') {
      const observer = new ResizeObserver(() => schedule(element));
      // Observe the fixed-width button. Watching the text itself would retrigger
      // the observer when fitting changes its line box.
      observer.observe(element.parentElement || element);
      cleanup = () => observer.disconnect();
    } else {
      const onResize = () => schedule(element);
      window.addEventListener('resize', onResize, { passive: true });
      cleanup = () => window.removeEventListener('resize', onResize);
    }
    observers.set(element, cleanup);
    element.__rankingFitText = element.textContent;
    schedule(element);
    document.fonts?.ready.then(() => { if (observers.has(element)) schedule(element); });
  },
  updated: scheduleWhenTextChanges,
  beforeUnmount(element) {
    cancelAnimationFrame(element.__rankingFitFrame || 0);
    observers.get(element)?.();
    observers.delete(element);
    delete element.__rankingFitText;
  },
};
