export function settlementMetric(result, side, project = {}, lang = 'zh') {
  const zh = lang === 'zh';
  const ref = project.project_ref || '';
  const reason = result?.reason || '';
  const race = ref.includes('-race-') || reason.startsWith('race_');
  const label = race ? (zh ? '完成用时' : 'TIME') : ref.includes('cargo-transport') || reason === 'delivered_cargo'
    ? (zh ? '送出数量' : 'DELIVERED') : ref.includes('dice-wall') || reason === 'board_sum'
      ? (zh ? '盘面和' : 'BOARD SUM') : (zh ? '得分' : 'SCORE');
  if (!result || reason === 'late_forfeit') return { label, value: '—', note: zh ? '未开赛' : 'NOT PLAYED' };
  const outcome = result[`${side}_outcome`];
  const conceded = reason === `${side}_surrendered` || outcome === 'surrendered';
  const expired = reason === `${side}_clock_expired` || reason === 'both_clocks_expired';
  const note = result.corrected ? (zh ? '裁判修正' : 'ADJUDICATED') : conceded ? (zh ? '认输' : 'CONCEDED') : expired ? (zh ? '包干时间耗尽' : 'TEAM TIME EXPIRED') : '';
  if (race && !result.corrected) {
    if (outcome !== 'target_reached') return { label, value: 'DNF', note: note || (zh ? '未完成' : 'NOT FINISHED') };
    const elapsed = result[`${side}_elapsed_ms`];
    if (elapsed == null || !Number.isFinite(Number(elapsed))) return { label, value:'—', note: zh ? '用时未记录' : 'TIME UNAVAILABLE' };
    const ms = Math.max(0, Number(elapsed));
    return { label, value:`${String(Math.floor(ms/60000)).padStart(2,'0')}:${String(Math.floor(ms/1000)%60).padStart(2,'0')}.${String(Math.floor(ms/10)%100).padStart(2,'0')}`, note };
  }
  return { label: race ? (zh ? '裁定成绩' : 'OFFICIAL RESULT') : label,
    value: Number(result[`${side}_score`] ?? 0).toLocaleString(zh ? 'zh-CN' : 'en-US'), note };
}
