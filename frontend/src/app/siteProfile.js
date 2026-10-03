import { TAB_IDS } from './tabRegistry.js';

export const TABLE_TABS = Object.freeze({
  trainer: TAB_IDS.TRAINER, tester: TAB_IDS.TESTER, battle: TAB_IDS.BATTLE,
  replay: TAB_IDS.REPLAY, analysis: TAB_IDS.ANALYSIS,
});
export const TAB_ROUTES = Object.freeze({ ...TABLE_TABS, settings: TAB_IDS.SETTINGS,
  gamer: TAB_IDS.GAMER, minigames: TAB_IDS.MINIGAMES, help: TAB_IDS.HELP,
  contact: TAB_IDS.CONTACT, admin: TAB_IDS.ADMIN, leaderboards: TAB_IDS.LEADERBOARDS });
export const isTableTab = tab => Object.values(TABLE_TABS).includes(tab);
export function siteProfile(location, entry = '') {
  return entry === 'tables' || location.hostname === 'tables.2048tables.online'
    || /^\/tables(?:\/|$)/.test(location.pathname) ? 'tables' : 'main';
}
export const currentSite = typeof window === 'undefined' ? 'main'
  : siteProfile(window.location, document.documentElement.dataset.site);

export function siteUrl(site, location = window.location) {
  if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') {
    const url = new URL(site === 'tables' ? '/tables/' : '/', location.origin);
    const port = new URLSearchParams(location.search || '').get('backend_port');
    if (port) url.searchParams.set('backend_port', port);
    return url;
  }
  return new URL(site === 'tables' ? 'https://tables.2048tables.online/' : 'https://2048tables.online/');
}
export function tabUrl(tab, site = isTableTab(tab) ? 'tables' : 'main', location) {
  const url = siteUrl(site, location);
  const route = Object.keys(TAB_ROUTES).find(key => TAB_ROUTES[key] === tab);
  if (route) url.searchParams.set('tab', route);
  return url;
}
export const MAIN_HOME_ENTRIES = Object.freeze([
  { id: 'play', title: 'menu.play', description: 'menu.descriptions.playSite', icon: 'Gamepad2', href: 'https://play.2048tables.online/' },
  { id: 'tables', title: 'menu.tables', description: 'menu.descriptions.tables', icon: 'Layers', site: 'tables' },
  { id: 'minigames', title: 'menu.minigames', description: 'menu.descriptions.minigames', icon: 'Grid2X2', tab: TAB_IDS.MINIGAMES },
  { id: 'ai', title: 'menu.ai', description: 'menu.descriptions.play', icon: 'Cpu', tab: TAB_IDS.GAMER },
  { id: 'settings', title: 'menu.settings', description: 'menu.descriptions.settings', icon: 'Settings', tab: TAB_IDS.SETTINGS },
  { id: 'help', title: 'menu.help', description: 'menu.descriptions.help', icon: 'CircleHelp', tab: TAB_IDS.HELP },
]);
export const TABLES_HOME_ENTRIES = Object.freeze([
  { id: 'trainer', title: 'tabs.trainer', description: 'menu.descriptions.practice', icon: 'Target', tab: TAB_IDS.TRAINER },
  { id: 'tester', title: 'menu.test', description: 'menu.descriptions.test', icon: 'ClipboardCheck', tab: TAB_IDS.TESTER },
  { id: 'battle', title: 'menu.battle', description: 'menu.descriptions.battle', icon: 'Swords', tab: TAB_IDS.BATTLE },
  { id: 'replay', title: 'tabs.replay', description: 'menu.descriptions.tableReplay', icon: 'Clapperboard', tab: TAB_IDS.REPLAY },
  { id: 'analysis', title: 'tabs.analysis', description: 'menu.descriptions.analysis', icon: 'ChartNoAxesCombined', tab: TAB_IDS.ANALYSIS },
  { id: 'settings', title: 'menu.settings', description: 'menu.descriptions.settings', icon: 'Settings', tab: TAB_IDS.SETTINGS },
]);
