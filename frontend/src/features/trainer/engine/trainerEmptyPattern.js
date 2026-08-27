export const EMPTY_PATTERN_ID = '__empty__';
export const EMPTY_PATTERN_CATEGORY = '__empty__';

export const isEmptyTrainerPattern = (pattern) => pattern === EMPTY_PATTERN_ID;

export const shouldDeferTrainerBoardSync = (pattern) => isEmptyTrainerPattern(pattern);

export const withEmptyPatternGroup = (groups = []) => [
  {
    category: EMPTY_PATTERN_CATEGORY,
    patterns: [EMPTY_PATTERN_ID],
  },
  ...groups,
];

export const stripTrainerQueryPayload = (payload = {}) => {
  const nextPayload = { ...payload };
  delete nextPayload.query_id;
  delete nextPayload.query_reason;
  delete nextPayload.catalog_version;
  delete nextPayload.full_pattern;
  delete nextPayload.prefetch_rng;
  return nextPayload;
};
