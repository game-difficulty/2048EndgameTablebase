import { TAB_IDS } from './tabRegistry.js';

export const AUXILIARY_ENTRIES = {
  live: { id: 'live', title: 'menu.live', icon: 'Radio', group: 'watch', href: 'https://live.2048tables.online/' },
  replay: { id: 'replay', title: 'menu.openVerseReplay', icon: 'Clapperboard', group: 'watch', href: '/verse-replay/' },
  announcements: { id: 'announcements', title: 'announcements.title', icon: 'Megaphone', group: 'info', tab: TAB_IDS.ANNOUNCEMENTS },
  quota: { id: 'quota', title: 'billing.quotaGuide.open', icon: 'Coins', group: 'info', dialog: 'quota' },
  help: { id: 'help', title: 'tabs.help', icon: 'CircleHelp', group: 'info', tab: TAB_IDS.HELP },
  more: { id: 'more', title: 'menu.more', icon: 'Ellipsis', tab: TAB_IDS.MORE },
};
export const HOME_AUXILIARY_ENTRIES = ['announcements', 'replay', 'more'].map(id => AUXILIARY_ENTRIES[id]);
export const AUXILIARY_GROUPS = [
  { id: 'watch', title: 'menu.watchAndReplay' },
  { id: 'info', title: 'menu.siteInformation' },
].map(group => ({ ...group, entries: Object.values(AUXILIARY_ENTRIES).filter(entry => entry.group === group.id) }));
