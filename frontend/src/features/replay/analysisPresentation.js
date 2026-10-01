export function formatAnalysisFit(value) {
  if (value == null || value === '' || !Number.isFinite(Number(value))) return '—';
  return `${(Number(value) * 100).toFixed(1)}%`;
}

export function replayDisplayName(value, fallback) {
  const name = String(value || '').split(/[\\/]/).pop();
  const internal = /^(?:\d+[_-])?(?:[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\.(?:vrs|rpl|txt)$/i;
  return name && !internal.test(name) ? name : fallback;
}

export function analysisScoreLabel(score, language = 'zh') {
  if (score == null || score === '' || !Number.isFinite(Number(score))) return '';
  const en = String(language).startsWith('en');
  const value = new Intl.NumberFormat(en ? 'en-US' : 'zh-CN').format(Number(score));
  return en ? `${value} pts` : `${value} 分`;
}
