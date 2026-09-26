export const TERMINAL_OVERLAY_DELAY_MS = 2000;

export function createTerminalOverlay({
  delay = TERMINAL_OVERLAY_DELAY_MS,
  setTimer = setTimeout,
  clearTimer = clearTimeout,
  onVisible = () => {},
} = {}) {
  let timer = null;
  let currentRunId = '';
  let dismissedRunId = '';

  function cancel() {
    if (timer !== null) clearTimer(timer);
    timer = null;
  }

  function update(runId, ended) {
    cancel();
    currentRunId = String(runId || '');
    onVisible(false);
    if (!ended || !currentRunId || dismissedRunId === currentRunId) return;
    const expectedRunId = currentRunId;
    timer = setTimer(() => {
      timer = null;
      if (currentRunId === expectedRunId && dismissedRunId !== expectedRunId) onVisible(true);
    }, delay);
  }

  function dismiss(runId = currentRunId) {
    if (runId) dismissedRunId = String(runId);
    cancel();
    onVisible(false);
  }

  function dispose() {
    cancel();
    onVisible(false);
  }

  return { update, dismiss, dispose };
}
