import { inject, provide } from 'vue';
const key = Symbol('room-activities');
export function provideActivities(context) { provide(key, context); }
export function useActivities() {
  const context = inject(key);
  if (!context) throw Error('Room activity provider required');
  return context;
}
