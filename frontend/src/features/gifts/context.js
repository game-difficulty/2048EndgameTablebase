import { inject, provide } from 'vue';
const giftClientKey = Symbol('gift-client');
export function provideGiftClient(client) { provide(giftClientKey, client); }
export function useGiftClient() {
  const client = inject(giftClientKey);
  if (!client) throw new Error('A gift host must provide its authenticated API and target');
  return client;
}
