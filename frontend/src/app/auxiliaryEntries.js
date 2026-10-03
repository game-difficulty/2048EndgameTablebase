import { TAB_IDS } from './tabRegistry.js';

export const AUXILIARY_ENTRIES = {
  live: { id: 'live', title: 'menu.live', icon: 'Radio', group: 'watch', href: 'https://live.2048tables.online/lobby' },
  tournament: { id: 'tournament', title: 'menu.tournament', icon: 'Trophy', group: 'watch', href: 'https://tournament.2048tables.online/' },
  replay: { id: 'replay', title: 'menu.openVerseReplay', icon: 'Clapperboard', group: 'watch', href: 'https://2048tables.online/verse-replay/' },
  announcements: { id: 'announcements', title: 'announcements.title', icon: 'Megaphone', group: 'info', tab: TAB_IDS.ANNOUNCEMENTS },
  leaderboards: { id: 'leaderboards', title: 'menu.openLeaderboards', icon: 'Trophy', tab: TAB_IDS.LEADERBOARDS },
  quota: { id: 'quota', title: 'billing.quotaGuide.open', icon: 'Coins', group: 'info', dialog: 'quota' },
  help: { id: 'help', title: 'tabs.help', icon: 'CircleHelp', group: 'info', tab: TAB_IDS.HELP },
  more: { id: 'more', title: 'menu.more', icon: 'Ellipsis', tab: TAB_IDS.MORE },
  contact: { id: 'contact', title: 'menu.contact', icon: 'Mail', tab: TAB_IDS.CONTACT },
  github: { id: 'github', title: 'menu.githubNote', icon: 'Github', iconOnly: true, href: 'https://github.com/game-difficulty/2048EndgameTablebase' },
};
export const HOME_AUXILIARY_ENTRIES = ['announcements', 'replay', 'leaderboards', 'more', 'contact', 'github'].map(id => AUXILIARY_ENTRIES[id]);
export const AUXILIARY_GROUPS = [
  { id: 'watch', title: 'menu.watchAndReplay' },
  { id: 'info', title: 'menu.siteInformation' },
].map(group => ({ ...group, entries: Object.values(AUXILIARY_ENTRIES).filter(entry => entry.group === group.id) }));
