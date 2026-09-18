export const TAB_IDS = {
  MAIN_MENU: 'MainMenuView',
  GAMER: 'GamerView',
  TRAINER: 'TrainerView',
  TESTER: 'TesterView',
  MINIGAMES: 'MinigamesView',
  BATTLE: 'BattleView',
  LEADERBOARDS: 'LeaderboardsView',
  REPLAY: 'ReplayReviewView',
  SETTINGS: 'SettingsView',
  HELP: 'HelpView',
  ADMIN: 'AdminView',
  ANNOUNCEMENTS: 'AnnouncementsView',
  MORE: 'MoreView',
};

export const MAIN_TAB_ID = TAB_IDS.MAIN_MENU;

export const TAB_ORDER = [
  TAB_IDS.MAIN_MENU,
  TAB_IDS.GAMER,
  TAB_IDS.TRAINER,
  TAB_IDS.TESTER,
  TAB_IDS.MINIGAMES,
  TAB_IDS.BATTLE,
  TAB_IDS.LEADERBOARDS,
  TAB_IDS.REPLAY,
  TAB_IDS.SETTINGS,
  TAB_IDS.HELP,
  TAB_IDS.ADMIN,
  TAB_IDS.ANNOUNCEMENTS,
  TAB_IDS.MORE,
];

export const TAB_REGISTRY = {
  [TAB_IDS.MORE]: { id: TAB_IDS.MORE, titleKey: 'menu.more', closable: true },
  [TAB_IDS.ANNOUNCEMENTS]: {
    id: TAB_IDS.ANNOUNCEMENTS,
    titleKey: 'announcements.title',
    closable: true,
  },
  [TAB_IDS.MAIN_MENU]: {
    id: TAB_IDS.MAIN_MENU,
    titleKey: 'tabs.home',
    closable: false,
  },
  [TAB_IDS.GAMER]: {
    id: TAB_IDS.GAMER,
    titleKey: 'tabs.gamer',
    closable: true,
  },
  [TAB_IDS.TRAINER]: {
    id: TAB_IDS.TRAINER,
    titleKey: 'tabs.trainer',
    closable: true,
  },
  [TAB_IDS.TESTER]: {
    id: TAB_IDS.TESTER,
    titleKey: 'tabs.tester',
    closable: true,
  },
  [TAB_IDS.MINIGAMES]: {
    id: TAB_IDS.MINIGAMES,
    titleKey: 'tabs.minigames',
    closable: true,
  },
  [TAB_IDS.BATTLE]: {
    id: TAB_IDS.BATTLE,
    titleKey: 'tabs.battle',
    closable: true,
  },
  [TAB_IDS.LEADERBOARDS]: {
    id: TAB_IDS.LEADERBOARDS,
    titleKey: 'tabs.leaderboards',
    closable: true,
  },
  [TAB_IDS.REPLAY]: {
    id: TAB_IDS.REPLAY,
    titleKey: 'tabs.replay',
    closable: true,
  },
  [TAB_IDS.SETTINGS]: {
    id: TAB_IDS.SETTINGS,
    titleKey: 'tabs.settings',
    closable: true,
  },
  [TAB_IDS.HELP]: {
    id: TAB_IDS.HELP,
    titleKey: 'tabs.help',
    closable: true,
  },
  [TAB_IDS.ADMIN]: {
    id: TAB_IDS.ADMIN,
    titleKey: 'tabs.admin',
    closable: true,
  },
};
