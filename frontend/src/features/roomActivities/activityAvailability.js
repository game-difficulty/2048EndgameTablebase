export function predictionIsOpen(market, now, connected, online) {
  return !!(connected && online && market?.status === 'open' && market.deadline > now);
}

export function envelopeIsClaimable(envelope, now, connected, userId, claimed = false) {
  return !!(connected && envelope?.status === 'active' && envelope.expires_at > now
    && envelope.claimed < envelope.count && envelope.sender_id !== userId && !claimed);
}
