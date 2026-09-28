import light01 from '../assets/project-icons/01-light.png?url';
import dark01 from '../assets/project-icons/01-dark.png?url';
import light02 from '../assets/project-icons/02-light.png?url';
import dark02 from '../assets/project-icons/02-dark.png?url';
import light03 from '../assets/project-icons/03-light.png?url';
import dark03 from '../assets/project-icons/03-dark.png?url';
import light04 from '../assets/project-icons/04-light.png?url';
import dark04 from '../assets/project-icons/04-dark.png?url';
import light04Animated from '../assets/project-icons/04-light.webp?url';
import dark04Animated from '../assets/project-icons/04-dark.webp?url';
import light05 from '../assets/project-icons/05-light.png?url';
import dark05 from '../assets/project-icons/05-dark.png?url';
import light06 from '../assets/project-icons/06-light.png?url';
import dark06 from '../assets/project-icons/06-dark.png?url';
import light07 from '../assets/project-icons/07-light.png?url';
import dark07 from '../assets/project-icons/07-dark.png?url';
import light08 from '../assets/project-icons/08-light.png?url';
import dark08 from '../assets/project-icons/08-dark.png?url';
import light09 from '../assets/project-icons/09-light.png?url';
import dark09 from '../assets/project-icons/09-dark.png?url';
import light10 from '../assets/project-icons/10-light.png?url';
import dark10 from '../assets/project-icons/10-dark.png?url';
import light11 from '../assets/project-icons/11-light.png?url';
import dark11 from '../assets/project-icons/11-dark.png?url';
import light12 from '../assets/project-icons/12-light.png?url';
import dark12 from '../assets/project-icons/12-dark.png?url';

const lightIcons = [light01, light02, light03, light04, light05, light06, light07, light08, light09, light10, light11, light12];
const darkIcons = [dark01, dark02, dark03, dark04, dark05, dark06, dark07, dark08, dark09, dark10, dark11, dark12];
const refs = [
  'tournament-cargo-transport-4x4',
  'tournament-spawn4-50-3x3',
  'tournament-evil-spawn-4x4',
  'tournament-pure2-full-race-3x3',
  'tournament-grand-full-undo-race-3x3',
  'tournament-dice-wall-3x3',
  'tournament-mirror-64x10-race-4x4',
  'tournament-256-brick-5x5',
  'tournament-isolated-island-hard-4x4',
  'tournament-shape-shifter-hard-12',
  'practice-hundred-step-seal-4x4',
  'practice-growing-tiles-4x4',
];

const lightByRef = Object.fromEntries(refs.map((ref, index) => [ref, lightIcons[index]]));
const darkByRef = Object.fromEntries(refs.map((ref, index) => [ref, darkIcons[index]]));

export function projectIconUrl(projectRef, theme = 'auto') {
  const dark = theme === 'dark' || (theme === 'auto' && typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches);
  if (projectRef === refs[3] && typeof window !== 'undefined' && !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
    return dark ? dark04Animated : light04Animated;
  }
  return (dark ? darkByRef : lightByRef)[projectRef] || null;
}
