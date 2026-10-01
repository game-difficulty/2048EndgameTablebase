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

// Only show rule-relevant public information; never infer hidden engine timers.
export function projectRuleMetrics(view, lang = 'zh', project = {}) {
  const p = view?.payload || {};
  const zh = lang === 'zh';
  const metrics = [];
  const steps = value => `${value} ${zh ? '步' : 'moves'}`;
  if (p.next_seal_in != null) {
    metrics.push({ key: 'seal', label: zh ? '封锁轮换剩余' : 'UNTIL SEAL ROTATION', value: steps(p.next_seal_in), warning: p.next_seal_in <= 10 });
  }
  const id = project.project_ref || project.id || p.projectId;
  const limit = p.tile_limit ?? project.tileLimit ?? (id === 'practice-full-load-4x4' ? 12 : null);
  if (limit != null && Array.isArray(p.board)) {
    const count = p.board.flat().filter(value => typeof value === 'number' && value > 0).length;
    metrics.push({ key: 'capacity', label: zh ? '方块数量 / 上限' : 'TILES / LIMIT', value: `${count} / ${limit}`, warning: count >= limit });
  }
  if (p.target_sum != null) {
    metrics.push({ key: 'target', label: zh ? '目标盘面和' : 'TARGET SUM', value: Number(p.target_sum).toLocaleString(zh ? 'zh-CN' : 'en-US') });
  } else if (p.target_count != null && p.target_tile != null) {
    metrics.push({ key: 'target', label: zh ? `${p.target_tile}砖目标` : `${p.target_tile} TILE TARGET`, value: String(p.target_count) });
  }
  if (view?.view_protocol === 'cargo-transport-v1' && p.move_count != null && p.move_count < 10 && !p.cargo) {
    metrics.push({ key: 'opening', label: zh ? '首个特殊块还剩' : 'UNTIL FIRST CARGO', value: steps(10 - p.move_count) });
  }
  return metrics;
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
