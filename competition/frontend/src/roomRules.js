export function ruleSummary(rules, poolSize = 32) {
  const steps = rules.steps || [];
  const games = 1 + steps.reduce((n, step) => n + Number(step.picks), 0);
  const minimum = games + steps.reduce((n, step) => n + Number(step.bans), 0);
  const integers = steps.every(s => ['first','second'].includes(s.actor) && ['picks','bans'].every(k => Number.isInteger(s[k]) && s[k] >= 0 && s[k] <= 15) && s.picks+s.bans>0);
  return { games, minimum, valid: steps.length > 0 && steps.length <= 24 && integers && games <= 15 && games % 2 === 1 && minimum <= poolSize
    && Number.isInteger(rules.team_size) && rules.team_size >= 1 && rules.team_size <= 16
    && ['unique','everyone','balanced','free'].includes(rules.lineup_policy)
    && (rules.lineup_policy !== 'unique' || rules.team_size >= games)
    && [['draft_seconds',5,3600],['lineup_seconds',5,3600],['team_clock_seconds',30,86400]].every(([k,min,max]) => Number.isInteger(rules[k]) && rules[k]>=min && rules[k]<=max) };
}
export function ruleRequest(rules) {
  const {preset,team_size,series_mode,lineup_policy,final_selection,draft_seconds,lineup_seconds,team_clock_seconds,steps} = rules;
  return {preset,team_size,series_mode,lineup_policy,final_selection,draft_seconds,lineup_seconds,team_clock_seconds,...(preset==='custom'?{steps}: {})};
}
export function lineupValid(rules, choices) {
  const keys = rules?.game_keys || ['A','B','C'], size = rules?.team_size || 3;
  if (Object.keys(choices).length !== keys.length || keys.some(k => !Number.isInteger(choices[k]) || choices[k]<1 || choices[k]>size)) return false;
  const counts = Array.from({length:size},(_,i)=>Object.values(choices).filter(p=>p===i+1).length);
  switch (rules?.lineup_policy || 'unique') {
    case 'free': return true;
    case 'everyone': return counts.filter(n=>n>0).length===Math.min(size,keys.length);
    case 'balanced': return Math.max(...counts)-Math.min(...counts)<=1;
    case 'unique': return Math.max(...counts)<=1;
    default: return false;
  }
}

export function lineupPolicyDescription(policy, lang = 'zh') {
  const descriptions = {
    unique: ['每人最多出场一次。', 'At most one game per player.'],
    everyone: ['允许重复，尽可能全员出场：局数足够时每人至少一局，否则每局安排不同选手；不要求次数平均。', 'Repeats allowed; use every player if there are enough games, otherwise use a different player in each game. Counts need not be balanced.'],
    balanced: ['允许重复，所有选手的出场次数（含零次）相差不超过 1。', 'Repeats allowed; appearance counts across all players (including zero) may differ by at most one.'],
    free: ['允许重复，不限制出场人数或次数。', 'Repeats allowed, with no participation or appearance-count limits.'],
  };
  return descriptions[policy || 'unique']?.[lang==='zh'?0:1] || '';
}
