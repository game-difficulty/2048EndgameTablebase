export const emptyHistoryFilters = () => ({ minScore: '', source: 'all', timeField: 'ended', from: '', to: '' });
export function historyFilterParams(f) {
  const result = { source: f.source, time_field: f.timeField };
  if (f.minScore !== '') {
    const score = Number(f.minScore);
    if (!Number.isSafeInteger(score) || score < 0) throw new Error('invalid');
    result.min_score = String(score);
  }
  const day = value => {
    const [y,m,d] = value.split('-').map(Number), date = new Date(y,m-1,d);
    if (!Number.isFinite(date.getTime()) || date.getFullYear() !== y || date.getMonth() !== m-1 || date.getDate() !== d) throw new Error('invalid');
    return date;
  };
  if (f.from) result.time_from = String(day(f.from).getTime()/1000);
  if (f.to) { const end = day(f.to); end.setDate(end.getDate()+1); result.time_to = String(end.getTime()/1000); }
  if (result.time_from && result.time_to && Number(result.time_from) >= Number(result.time_to)) throw new Error('invalid');
  return result;
}

export function completeHistoryDates(filters, changed) {
  const other = changed === 'from' ? 'to' : 'from';
  if (filters[changed] && !filters[other]) filters[other] = filters[changed];
}
export function historyPageTarget(value, count) {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || String(value).trim() === '') return null;
  return Math.max(1, Math.min(number, count));
}
