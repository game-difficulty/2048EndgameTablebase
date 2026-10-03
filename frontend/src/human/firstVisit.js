import { ref } from 'vue';
// Shared state lifecycle; individual flows keep their own presentation.
export function createFirstVisit(request) {
  const pending = ref(null), busy = ref(false), error = ref('');
  let account = null, generation = 0;
  function reset(id) { account = id; generation++; pending.value = id ? { id: 'play_rules', version: 1 } : null; error.value = ''; busy.value = false; }
  function apply(rows) { pending.value = rows.find(row => row.required && !row.status) || null; }
  async function confirm() {
    if (busy.value || !pending.value || !account) return;
    const serial = generation, flow = pending.value;
    busy.value = true; error.value = '';
    try {
      const result = await request(`/api/human/me/first-visit/${flow.id}`, { method: 'POST', body: {version: flow.version, status: 'acknowledged'} });
      if (serial === generation) apply(result.first_visit);
    } catch { if (serial === generation) error.value = '确认未能保存，请联网后重试。'; }
    finally { if (serial === generation) busy.value = false; }
  }
  return { pending, busy, error, reset, apply, confirm };
}
