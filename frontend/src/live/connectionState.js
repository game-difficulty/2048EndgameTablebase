export function liveConnectionState({ connected, synchronized, seenSnapshot, online }) {
  if (!connected || !synchronized) return seenSnapshot ? 'reconnecting' : 'loading';
  return online ? 'live' : 'offline';
}
