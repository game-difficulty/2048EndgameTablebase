export function battleCountdownState({
  deadline,
  now = Date.now(),
  status = '',
  correcting = false,
  resolving = false,
  pausedSeconds = 90,
} = {}) {
  if (resolving && status === 'playing' && !correcting) {
    return { seconds: null, urgent: false, critical: false, paused: true };
  }
  if (correcting && status === 'playing') {
    return {
      seconds: Math.max(0, Number(pausedSeconds) || 0),
      urgent: false,
      critical: false,
      paused: true,
    };
  }
  const deadlineMs = Date.parse(String(deadline || ''));
  if (status !== 'playing' || !Number.isFinite(deadlineMs)) {
    return { seconds: null, urgent: false, critical: false, paused: false };
  }
  const seconds = Math.max(0, Math.ceil((deadlineMs - Number(now)) / 1000));
  return {
    seconds,
    urgent: seconds > 0 && seconds <= 5,
    critical: seconds > 0 && seconds <= 2,
    paused: false,
  };
}
