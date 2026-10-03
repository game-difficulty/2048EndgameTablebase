export function marketTitle(kind, lang='zh') {
  const labels={winner:['全场胜方','Match winner'],first_two:['前两局比分 · 黄 : 白','First two games · Yellow : White'],
    clinch_3:['首次达到 3 胜时的比分 · 黄 : 白','Score at first 3 wins · Yellow : White'],
    clinch_4:['首次达到 4 胜时的比分 · 黄 : 白','Score at first 4 wins · Yellow : White']};
  return (labels[kind] || [kind,kind])[lang==='zh'?0:1];
}
