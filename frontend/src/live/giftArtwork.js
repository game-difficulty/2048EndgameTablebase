import { liveSupporterLevel } from './supporterIdentity.js';

export const referenceArtwork = Object.freeze({
  knowledge: 'knowledge-cutout', meaning: 'meaning', whale: 'whale', moai: 'moai-cutout',
  button: 'button', tea: 'tea', chicken: 'chicken', serious: 'serious',
});
export const ceremonyGifts = new Set(['final', '2048', 'crown', 'legend']);
const supporterMotion = new Set(['button', 'whale', 'moai', 'tea', 'chicken', 'serious', 'rip']);
export function giftAnimation(event) {
  if (event?.type !== 'gift') return null;
  if (ceremonyGifts.has(event.gift_id)) return 'ceremony';
  return supporterMotion.has(event.gift_id) && liveSupporterLevel(event.actor) > 0 ? 'supporter' : null;
}
export const giftAsset = name => `/live-gifts/${name}.webp`;

export function giftChat(event) {
  return { ...event, ...event.actor, id: `gift:${event.combo_id || event.id}` };
}

// Combo updates keep their original place in the chat, including HTTP/WS races.
export function mergeLiveChat(history, incoming) {
  const messages = new Map(history.map(item => [item.id, item]));
  for (const event of incoming) {
    const item = event.type === 'gift' ? giftChat(event) : event;
    const previous = messages.get(item.id);
    if (previous && item.type === 'gift') {
      if (item.combo_count > previous.combo_count) messages.set(item.id, { ...item, at: previous.at });
    } else messages.set(item.id, item);
  }
  return [...messages.values()].sort((a, b) => a.at - b.at).slice(-100);
}
