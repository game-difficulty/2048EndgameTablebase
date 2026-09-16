export function liveConnectionState({ connected, synchronized, seenSnapshot, online, paused }) {
  if (!connected || !synchronized) return seenSnapshot ? 'reconnecting' : 'loading';
  if (paused) return 'paused';
  return online ? 'live' : 'offline';
}
