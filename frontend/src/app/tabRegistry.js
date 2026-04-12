export const TAB_IDS = {
  MAIN_MENU: 'MainMenuView',
  GAMER: 'GamerView',
  TRAINER: 'TrainerView',
  TESTER: 'TesterView',
  MINIGAMES: 'MinigamesView',
  REPLAY: 'ReplayReviewView',
  NOTEBOOK: 'NotebookView',
  SETTINGS: 'SettingsView',
  HELP: 'HelpView',
};

export const MAIN_TAB_ID = TAB_IDS.MAIN_MENU;

export const TAB_ORDER = [
  TAB_IDS.MAIN_MENU,
  TAB_IDS.GAMER,
  TAB_IDS.TRAINER,
  TAB_IDS.TESTER,
  TAB_IDS.MINIGAMES,
  TAB_IDS.REPLAY,
  TAB_IDS.NOTEBOOK,
  TAB_IDS.SETTINGS,
  TAB_IDS.HELP,
];

export const TAB_REGISTRY = {
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
  [TAB_IDS.REPLAY]: {
    id: TAB_IDS.REPLAY,
    titleKey: 'tabs.replay',
    closable: true,
  },
  [TAB_IDS.NOTEBOOK]: {
    id: TAB_IDS.NOTEBOOK,
    titleKey: 'tabs.notebook',
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
};
