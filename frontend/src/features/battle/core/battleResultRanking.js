const isCompleted = (result) => String(result?.status || '') === 'completed';
const isLiveRanked = (result) => (
  ['playing', 'completed', 'disconnected'].includes(String(result?.status || ''))
);
const goodness = (result) => Number(result?.goodness_of_fit ?? 0);

export function rankBattleResults(results, { mode = 'final' } = {}) {
  const eligible = mode === 'live' ? isLiveRanked : isCompleted;
  const ordered = (results || []).map((result, originalIndex) => ({
    ...result,
    originalIndex,
  })).sort((left, right) => {
    const eligibilityOrder = Number(eligible(right)) - Number(eligible(left));
    if (eligibilityOrder) return eligibilityOrder;
    if (!eligible(left)) return left.originalIndex - right.originalIndex;
    return goodness(right) - goodness(left) || left.originalIndex - right.originalIndex;
  });

  let previousGoodness = null;
  let previousRank = null;
  let rankedCount = 0;
  return ordered.map((result, index) => {
    let rank = null;
    if (eligible(result)) {
      rankedCount += 1;
      const currentGoodness = goodness(result);
      rank = previousGoodness === currentGoodness ? previousRank : rankedCount;
      previousGoodness = currentGoodness;
      previousRank = rank;
    }
    const { originalIndex: _originalIndex, ...publicResult } = result;
    return { ...publicResult, rank };
  });
}

export function isBattleResultDraw(rankedResults, { mode = 'final' } = {}) {
  if (mode !== 'final') return false;
  const first = rankedResults?.[0];
  const second = rankedResults?.[1];
  return Boolean(
    isCompleted(first)
    && isCompleted(second)
    && goodness(first) === goodness(second),
  );
}
