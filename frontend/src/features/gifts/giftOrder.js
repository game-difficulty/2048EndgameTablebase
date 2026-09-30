export const DEFAULT_FEATURED_GIFT_IDS = Object.freeze([
  'two', 'four', 'heart', 'flowers', 'moai', 'button', 'tea', 'whale', 'rip',
]);

function idsFromCatalog(catalog) {
  return (Array.isArray(catalog) ? catalog : [])
    .map(item => typeof item === 'string' ? item : item?.id)
    .filter(id => typeof id === 'string' && id);
}

export function defaultGiftOrder(catalog) {
  const ids = idsFromCatalog(catalog);
  const known = new Set(ids);
  return [...DEFAULT_FEATURED_GIFT_IDS.filter(id => known.has(id)),
    ...ids.filter(id => !DEFAULT_FEATURED_GIFT_IDS.includes(id))];
}

export function normalizeGiftOrder(saved, catalog) {
  const fallback = defaultGiftOrder(catalog);
  const known = new Set(fallback);
  const result = [];
  if (Array.isArray(saved)) {
    for (const id of saved) {
      if (known.has(id) && !result.includes(id)) result.push(id);
    }
  }
  for (const id of fallback) if (!result.includes(id)) result.push(id);
  return result;
}

export function moveGift(order, giftId, targetIndex) {
  const result = [...order];
  const from = result.indexOf(giftId);
  if (from < 0 || !result.length) return result;
  const [item] = result.splice(from, 1);
  const target = Math.max(0, Math.min(result.length, Number(targetIndex) || 0));
  result.splice(target, 0, item);
  return result;
}
