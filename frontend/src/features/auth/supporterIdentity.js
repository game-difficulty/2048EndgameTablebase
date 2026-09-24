export function liveSupporterLevel(actor) {
  if (actor?.supporter_level === 2) return 2;
  return actor?.supporter_level === 1 || actor?.supporter ? 1 : 0;
}

export function entranceChat(event) {
  return { type: 'entrance', id: event.id, at: event.at, ...event.actor };
}

export function giftEffectDuration(event) {
  if (event.type === 'entrance') return 5000;
  const base = event.tier >= 3 ? 5000 : event.tier >= 1 ? 3500 : 2500;
  return base + (liveSupporterLevel(event.actor) === 2 ? 1500 : 0);
}
