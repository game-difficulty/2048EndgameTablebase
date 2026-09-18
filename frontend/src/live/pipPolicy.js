export function canConnectLive(hidden, pipActive) {
  return !hidden || pipActive;
}
export function backgroundExpired(hidden, pipActive, deadline, now) {
  return hidden && !pipActive && deadline > 0 && now >= deadline;
}
