export const createEmptyMinigameMenu = () => ({
  difficulty: 1,
  sections: [],
  currentGameId: '',
});

export const createEmptyMinigameState = () => ({
  gameId: '',
  title: '',
  difficulty: 1,
  board: new Array(16).fill(0),
  shape: { rows: 4, cols: 4 },
  score: 0,
  best: 0,
  status: 'running',
  animation: {
    effects: [],
    pageEffects: [],
  },
  view: {
    hiddenMask: new Array(16).fill(false),
    blockedMask: new Array(16).fill(false),
    smallLabels: new Array(16).fill(''),
    tileOverlays: {},
    tileTextOverride: {},
    tileStyleVariant: {},
    coverSprites: {},
  },
  hud: {
    score: 0,
    best: 0,
    infoText: '',
    customPanels: [],
  },
  powerups: {
    enabled: false,
    counts: { bomb: 0, glove: 0, twist: 0 },
    activeMode: null,
  },
  interaction: {
    active: false,
    mode: null,
    targetType: null,
    phase: null,
    validTargets: [],
    selectedIndices: [],
    hintKey: '',
  },
  messages: {},
});
