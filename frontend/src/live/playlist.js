export const BUILTIN_TRACKS = [
  { id: 'synthwave', title: 'Synthwave 15k', artist: 'The Cynic Project', src: '/live-music/synthwave-15k-v1.mp3', source: 'https://opengameart.org/content/calm-ambient-2-synthwave-15k' },
  { id: 'persistence', title: 'Persistence', artist: 'James Gargette', src: '/live-music/persistence-v1.mp3', source: 'https://opengameart.org/content/persistence' },
];

export function nextTrackIndex(index, length, delta = 1) {
  return length ? ((index + delta) % length + length) % length : 0;
}

export function audioFileKey(file) {
  return `${file.name}:${file.size}:${file.lastModified}`;
}
