import { ref } from 'vue';
import { useActivities } from './context.js';

// Dismissing a card only hides this room's activity, not its other entry points.
export function useActivityDismissals(kind) {
  const { room } = useActivities();
  const key = `room:${room.id}:activity:${kind}:dismissed:v1`;
  const dismissed = ref([]);
  try {
    const stored = JSON.parse(sessionStorage.getItem(key) || '[]');
    if (Array.isArray(stored)) dismissed.value = stored.filter(id => typeof id === 'string').slice(-30);
  } catch {}
  function dismiss(id) {
    dismissed.value = [...dismissed.value.filter(value => value !== id), id].slice(-30);
    try { sessionStorage.setItem(key, JSON.stringify(dismissed.value)); } catch {}
  }
  return { dismissed, dismiss };
}
