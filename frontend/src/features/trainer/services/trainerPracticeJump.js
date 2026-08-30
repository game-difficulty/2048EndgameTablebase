let pendingJumpDetail = null;
let jumpConsumer = null;
let practiceContext = null;
let contextConsumer = null;

function flushPendingJump() {
  if (!pendingJumpDetail || typeof jumpConsumer !== 'function') {
    return;
  }
  const detail = pendingJumpDetail;
  pendingJumpDetail = null;
  jumpConsumer(detail);
}

export function queueTrainerPracticeJump(detail) {
  if (!detail?.hex) {
    return false;
  }
  pendingJumpDetail = detail;
  practiceContext = detail?.context || null;
  contextConsumer?.(practiceContext);
  flushPendingJump();
  return true;
}

export function registerTrainerPracticeContextConsumer(consumer) {
  contextConsumer = typeof consumer === 'function' ? consumer : null;
  contextConsumer?.(practiceContext);
  return () => {
    if (contextConsumer === consumer) contextConsumer = null;
  };
}

export function clearTrainerPracticeContext(kind = '') {
  if (kind && practiceContext?.kind !== kind) return false;
  practiceContext = null;
  contextConsumer?.(null);
  return true;
}

export function registerTrainerPracticeJumpConsumer(consumer) {
  jumpConsumer = typeof consumer === 'function' ? consumer : null;
  flushPendingJump();

  return () => {
    if (jumpConsumer === consumer) {
      jumpConsumer = null;
    }
  };
}

export function resetTrainerPracticeJumpQueue() {
  pendingJumpDetail = null;
  jumpConsumer = null;
  practiceContext = null;
  contextConsumer = null;
}
