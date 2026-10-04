// Push revisions drive freshness; polling only covers a missed notification.
export function competitionPollInterval({ connected, hidden, opened, busy, pending, markets = [] }) {
  if (!connected || hidden || busy) return Infinity;
  if (pending) return 5000;
  if (opened && markets.some(m => m.status === 'open')) return 5000;
  if (opened && markets.some(m => !['settled', 'void'].includes(m.status))) return 15000;
  return 60000;
}
