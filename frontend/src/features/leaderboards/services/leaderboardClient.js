const readJson = async (response) => {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload?.detail || `HTTP ${response.status}`);
  }
  return payload;
};

export const fetchLeaderboardCatalog = async () => {
  const response = await fetch('/api/leaderboards', {
    headers: { Accept: 'application/json' },
  });
  return readJson(response);
};

export const fetchLeaderboard = async (boardKey, { limit } = {}) => {
  const query = Number.isInteger(limit) && limit > 0
    ? `?limit=${encodeURIComponent(limit)}`
    : '';
  const response = await fetch(`/api/leaderboards/${encodeURIComponent(boardKey)}${query}`, {
    headers: { Accept: 'application/json' },
    cache: 'no-store',
  });
  return readJson(response);
};
