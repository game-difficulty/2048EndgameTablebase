export function predictionIsOpen(market, now, connected, online) {
  return !!(connected && online && market?.status === 'open' && market.deadline > now);
}

// Activity state, not room content kind, determines the shared dock hint.
export function openPredictionMarket(state, now, connected, online) {
  if (!connected || !online || state?.available === false) return null;
  const markets = state?.markets || (state?.market ? [state.market] : []);
  return markets.find(market => market.status === 'open'
    && (market.deadline == null || market.deadline > now)) || null;
}

export function envelopeIsClaimable(envelope, now, connected, claimed = false) {
  return !!(connected && envelope?.status === 'active' && envelope.expires_at > now
    && envelope.claimed < envelope.count && !claimed);
}
