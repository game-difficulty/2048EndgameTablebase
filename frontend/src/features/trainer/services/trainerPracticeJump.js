let pendingJumpDetail = null;
let jumpConsumer = null;

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
  flushPendingJump();
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
