export function bagCaption(bag, now, lang) {
  if (bag.drawn_at != null) return lang === 'zh' ? '已开奖' : 'Results';
  const seconds = Math.max(0, Math.ceil(bag.draw_at - now));
  if (!seconds) return lang === 'zh' ? '开奖中' : 'Drawing';
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

export function mergeBags(previous, incoming, personal = false) {
  return incoming.map(bag => {
    if (personal) return { ...bag, resultKnown: true };
    const old = previous.find(item => item.id === bag.id);
    return { ...bag, joined: old?.joined, award: old?.award, present: old?.present,
      resultKnown: Boolean(old?.resultKnown && old.drawn_at === bag.drawn_at) };
  });
}

export function visibleBags(bags, now) {
  return bags.filter(bag => bag.drawn_at == null || bag.expires_at > now);
}
