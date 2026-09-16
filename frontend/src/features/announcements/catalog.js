import { TAB_IDS, TAB_REGISTRY } from '../../app/tabRegistry.js';

// Keep published IDs and copy revisions stable so archived notices remain readable.
export const ANNOUNCEMENTS = Object.freeze([
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
