export function activeBestScore(bests, run, variant) {
  const historical = Math.max(0, Number(bests?.[variant]) || 0);
  const current = run?.variant === variant ? Math.max(0, Number(run.score) || 0) : 0;
  return Math.max(historical, current);
}
