export function projectPerformanceMetric(view, lang = 'zh') {
  const payload = view?.payload || {};
  const chinese = lang === 'zh';
  let label;
  let value;

  if (view?.view_protocol === 'cargo-transport-v1') {
    label = chinese ? '已送出' : 'DELIVERED';
    value = payload.deliveries ?? payload.score;
  } else if (payload.target_count != null && payload.target_tile != null) {
    label = chinese ? `${payload.target_tile}砖数量` : `${payload.target_tile} TILES`;
    value = payload.current_target_count;
  } else if (payload.result_metric === 'board_sum' || payload.target_sum != null) {
    label = chinese ? '盘面和' : 'BOARD SUM';
    value = payload.board_sum;
  } else {
    label = chinese ? '得分' : 'SCORE';
    value = payload.score;
  }

  return {
    label,
    value: Number(value ?? 0).toLocaleString(chinese ? 'zh-CN' : 'en-US'),
  };
}

export function projectResultValue(result, side, lang = 'zh') {
  if (!result) return '—';
  const elapsed = result[`${side}_elapsed_ms`];
  if (!result.corrected && result.reason?.startsWith('race_') && elapsed != null) {
    if (result[`${side}_outcome`] !== 'target_reached') return lang === 'zh' ? '未达标' : 'NOT FINISHED';
    const ms = Math.max(0, Number(elapsed));
    return `${String(Math.floor(ms / 60000)).padStart(2, '0')}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')}.${String(Math.floor(ms / 10) % 100).padStart(2, '0')}`;
  }
  return Number(result[`${side}_score`] ?? 0).toLocaleString(lang === 'zh' ? 'zh-CN' : 'en-US');
}
