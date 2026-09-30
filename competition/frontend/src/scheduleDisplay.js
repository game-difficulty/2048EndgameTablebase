import { language } from './i18n.js';
export function scheduleTime(value) {
  return value ? new Intl.DateTimeFormat(language.value==='en'?'en-GB':'zh-CN', { timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(value)) : '';
}
export function roomMatchTitle(item) {
  return item.schedule ? `${item.schedule.yellow_name} VS ${item.schedule.white_name}` : item.name;
}
export function roomMatchScore(item) {
  return item.series_score && item.series_score.yellow !== undefined ? `${item.series_score.yellow}:${item.series_score.white}` : '';
}
