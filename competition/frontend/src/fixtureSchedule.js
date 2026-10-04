// datetime-local has no timezone. Treat its text as Beijing time regardless of
// browser/device timezone; never pass it through the host's local Date constructor.
export function beijingInput(value) {
  if (!value) return '';
  const date = new Date(new Date(value).getTime() + 8 * 3600000);
  return Number.isFinite(date.getTime()) ? date.toISOString().slice(0, 16) : '';
}

export function beijingISO(value) {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(value || '')) return null;
  const date = new Date(`${value}:00+08:00`);
  if (!Number.isFinite(date.getTime()) || beijingInput(date.toISOString()) !== value) return null;
  return date.toISOString();
}

export function fixtureStatus(item, lang = 'zh') {
  const label = (zh, en) => lang === 'zh' ? zh : en;
  if (item.status === 'UNSCHEDULED') return label('待约时间', 'Time not agreed');
  if (item.status === 'CANCELLED') return label('已取消', 'Cancelled');
  if (item.status === 'FINISHED') {
    return item.exception === 'both_late' ? label('双方未就位 · 0:0', 'Neither team ready · 0:0') : label('已结束', 'Finished');
  }
  if (['SEATING', 'READY_CHECK'].includes(item.status)) {
    return item.proposed_at ? label('改期待确认', 'Reschedule pending') : label('待开赛', 'Scheduled');
  }
  return label('比赛中', 'In progress');
}

export function stageGroups(teams, assignments, names) {
  return names.map((name, index) => ({
    name: name.trim(), team_ids: teams.filter(team => assignments[team.id] === index).map(team => team.id),
  }));
}
