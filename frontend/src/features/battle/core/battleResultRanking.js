const isCompleted = (result) => String(result?.status || '') === 'completed';
const isLiveRanked = (result) => (
  ['playing', 'completed', 'disconnected'].includes(String(result?.status || ''))
);
const goodness = (result) => Number(result?.goodness_of_fit ?? 0);

const freeRankingEligible = (result, mode) => (
  Boolean(result?.mode_data?.ranking_eligible)
  && (
    mode === 'live'
      ? ['playing', 'completed', 'disconnected'].includes(String(result?.status || ''))
      : String(result?.status || '') === 'completed'
  )
);

function rankFreeGoodness(results, { mode }) {
  const ordered = (results || []).map((result, originalIndex) => ({
    ...result,
    originalIndex,
    rankClass: Number(freeRankingEligible(result, mode)),
  })).sort((left, right) => {
    const eligibilityOrder = right.rankClass - left.rankClass;
    if (eligibilityOrder) return eligibilityOrder;
    if (!left.rankClass) return left.originalIndex - right.originalIndex;
    return goodness(right) - goodness(left) || left.originalIndex - right.originalIndex;
  });
  let rankedCount = 0;
  let previousKey = '';
  let previousRank = null;
  return ordered.map((result) => {
    let rank = null;
    if (result.rankClass > 0) {
      rankedCount += 1;
      const key = `${result.rankClass}:${goodness(result)}`;
      rank = key === previousKey ? previousRank : rankedCount;
      previousKey = key;
      previousRank = rank;
    }
    const { originalIndex: _originalIndex, rankClass: _rankClass, ...publicResult } = result;
    return { ...publicResult, rank };
  });
}

export function rankBattleResults(results, { mode = 'final', battleMode = 'goodness' } = {}) {
  if (battleMode === 'free_goodness') return rankFreeGoodness(results, { mode });
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

export function isBattleResultDraw(rankedResults, { mode = 'final', battleMode = 'goodness' } = {}) {
  if (mode !== 'final') return false;
  const first = rankedResults?.[0];
  const second = rankedResults?.[1];
  if (battleMode === 'free_goodness') {
    return Boolean(first?.rank === 1 && second?.rank === 1);
  }
  return Boolean(
    isCompleted(first)
    && isCompleted(second)
    && goodness(first) === goodness(second),
  );
}
