import { TAB_IDS, TAB_REGISTRY } from '../../app/tabRegistry.js';

// Keep published IDs and copy revisions stable so archived notices remain readable.
export const ANNOUNCEMENTS = Object.freeze([
  Object.freeze({
    id: '2026-10-03-tables-competition-analysis',
    date: '2026-10-03',
    titleKey: 'announcements.octoberUpdate.title',
    summaryKey: 'announcements.octoberUpdate.summary',
    bodyKeys: Object.freeze(['announcements.octoberUpdate.intro']),
    featureKeys: Object.freeze([
      'announcements.octoberUpdate.features.tables',
      'announcements.octoberUpdate.features.competition',
      'announcements.octoberUpdate.features.duels',
      'announcements.octoberUpdate.features.sumGoals',
      'announcements.octoberUpdate.features.analysis',
      'announcements.octoberUpdate.features.play',
      'announcements.octoberUpdate.features.replay',
      'announcements.octoberUpdate.features.live',
      'announcements.octoberUpdate.features.accounts',
    ]),
    noticeKey: 'announcements.octoberUpdate.notice',
    siteUrl: 'https://tables.2048tables.online/',
    links: Object.freeze([
      Object.freeze({ labelKey: 'announcements.octoberUpdate.links.tables', url: 'https://tables.2048tables.online/' }),
      Object.freeze({ labelKey: 'announcements.octoberUpdate.links.play', url: 'https://play.2048tables.online/' }),
      Object.freeze({ labelKey: 'announcements.octoberUpdate.links.events', url: 'https://tournament.2048tables.online/events' }),
      Object.freeze({ labelKey: 'announcements.octoberUpdate.links.duels', url: 'https://tournament.2048tables.online/duels' }),
      Object.freeze({ labelKey: 'announcements.octoberUpdate.links.timeAttack', url: 'https://tournament.2048tables.online/time-attacks' }),
      Object.freeze({ labelKey: 'announcements.octoberUpdate.links.live', url: 'https://live.2048tables.online/lobby' }),
    ]),
    target: Object.freeze({ type: 'announcement', id: '2026-10-03-tables-competition-analysis' }),
  }),
  Object.freeze({
    id: '2026-09-29-play-beta',
    date: '2026-09-29',
    titleKey: 'announcements.playBeta.title',
    summaryKey: 'announcements.playBeta.summary',
    bodyKeys: Object.freeze([
      'announcements.playBeta.intro',
    ]),
    featureKeys: Object.freeze([
      'announcements.playBeta.features.games',
      'announcements.playBeta.features.profile',
      'announcements.playBeta.features.analysis',
      'announcements.playBeta.features.migration',
      'announcements.playBeta.features.practice',
    ]),
    noticeKey: 'announcements.playBeta.notice',
    siteUrl: 'https://play.2048tables.online/',
    target: Object.freeze({ type: 'announcement', id: '2026-09-29-play-beta' }),
  }),
  Object.freeze({
    id: '2026-09-16-live-rewards',
    date: '2026-09-16',
    titleKey: 'announcements.liveRewards.title',
    summaryKey: 'announcements.liveRewards.summary',
    bodyKey: 'announcements.liveRewards.body',
    liveUrl: 'https://live.2048tables.online/',
    rewardCopyKey: 'billing.quotaGuide.earn',
    target: Object.freeze({ type: 'announcement', id: '2026-09-16-live-rewards' }),
  }),
]);

export const latestAnnouncement = ANNOUNCEMENTS[0];
export const DISMISSED_KEY = '2048tables:dismissed-announcement';
export const findAnnouncement = (id) => ANNOUNCEMENTS.find(item => item.id === id) || latestAnnouncement;

export function resolveAnnouncementTarget(target) {
  if (target?.type === 'tab' && TAB_REGISTRY[target.id] && target.id !== TAB_IDS.ADMIN) {
    return { tab: target.id };
  }
  return { tab: TAB_IDS.ANNOUNCEMENTS, announcementId: findAnnouncement(target?.id).id };
}

export function wasDismissed(storage, id) {
  try { return storage.getItem(DISMISSED_KEY) === id; } catch { return false; }
}

export function dismissAnnouncement(storage, id) {
  try { storage.setItem(DISMISSED_KEY, id); } catch { /* Session dismissal still works. */ }
}
