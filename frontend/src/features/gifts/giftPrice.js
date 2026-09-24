export function giftPrice(gift, quantity) {
  if (!gift || !Number.isInteger(quantity) || quantity < 1 || quantity > 1000) return null;
  if (gift.base_units == null) return gift.totals?.[quantity - 1] ?? null;
  const raw = BigInt(gift.base_units) * BigInt(quantity) * BigInt(gift.global_multiplier_units);
  const whole = raw / 1000n, rest = raw % 1000n;
  // Match Python round(): ties go to the even integer, not always upward.
  return Number(whole + (rest > 500n || (rest === 500n && whole % 2n === 1n) ? 1n : 0n));
}
