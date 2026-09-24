import { createGiftClient } from '../features/gifts/giftApi.js';
export { createGiftClient };
// Legacy public API for older consumers. New hosts provide their own client.
export const { giftApi, sendGift } = createGiftClient({ base: '/api/live/gifts', target: 'live:ai-classic' });
