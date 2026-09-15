export const likeReactionWeights = Object.freeze([
  ['heart', 25], ['flowers', 25], ['666', 20], ['two', 15],
  ['tea', 5], ['whale', 5], ['button', 3], ['moai', 2],
].map(entry => Object.freeze(entry)));

export function pickLikeReaction(random = Math.random) {
  let ticket = random() * 100;
  for (const [id, weight] of likeReactionWeights) {
    ticket -= weight;
    if (ticket < 0) return id;
  }
  return 'moai';
}
