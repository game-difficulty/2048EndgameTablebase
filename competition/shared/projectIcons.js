import icon01 from '../assets/project-icons/01.svg?url';
import icon02 from '../assets/project-icons/02.svg?url';
import icon03 from '../assets/project-icons/03.svg?url';
import icon04 from '../assets/project-icons/04.svg?url';
import icon05 from '../assets/project-icons/05.svg?url';
import icon06 from '../assets/project-icons/06.svg?url';
import icon07 from '../assets/project-icons/07.svg?url';
import icon08 from '../assets/project-icons/08.svg?url';
import icon09 from '../assets/project-icons/09.svg?url';
import icon10 from '../assets/project-icons/10.svg?url';
import icon11 from '../assets/project-icons/11.svg?url';
import icon12 from '../assets/project-icons/12.svg?url';

const icons = [icon01, icon02, icon03, icon04, icon05, icon06, icon07, icon08, icon09, icon10, icon11, icon12];
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

const byRef = Object.fromEntries(refs.map((ref, index) => [ref, icons[index]]));

export function projectIconUrl(projectRef) {
  return byRef[projectRef] || null;
}
