export const TRAINER_GUEST_QUERY_LIMIT_CODE = 'GUEST_QUERY_LIMIT_REACHED';
export const TRAINER_GUEST_QUERY_EXHAUSTED_CODE = 'GUEST_QUERY_ALLOWANCE_EXHAUSTED';
export const TRAINER_GUEST_NETWORK_LIMIT_CODE = 'GUEST_NETWORK_QUERY_LIMIT_REACHED';
export const TRAINER_GUEST_TABLE_LOCKED_CODE = 'GUEST_TABLE_NOT_AVAILABLE';
export const TRAINER_GUEST_TABLE_LOGIN_REQUIRED_CODE = 'GUEST_TABLE_LOGIN_REQUIRED';

export function isRegisteredTrainerActor(actor) {
  return actor?.kind === 'user';
}

export function isGuestTrainerActor(actor) {
  return actor?.kind === 'guest';
}

export function normalizeTrainerGuestAllowance(rawAllowance, fallback = null) {
  if (!rawAllowance || typeof rawAllowance !== 'object') return fallback;
  const remaining = Number(rawAllowance.remaining);
  const total = Number(rawAllowance.total ?? rawAllowance.limit);
  if (!Number.isFinite(remaining) || !Number.isFinite(total)) return fallback;
  return {
    remaining: Math.max(0, Math.trunc(remaining)),
    total: Math.max(0, Math.trunc(total)),
  };
}

export function tablebaseFullPattern(table = {}) {
  return String(
    table.fullPattern
      || table.full_pattern
      || (table.pattern && table.target ? `${table.pattern}_${table.target}` : ''),
  );
}

export function isTrainerTableGuestAvailable(tables = [], fullPattern = '') {
  const normalizedPattern = String(fullPattern || '');
  const table = tables.find((entry) => tablebaseFullPattern(entry) === normalizedPattern);
  if (!table) return false;
  return table.guestAvailable === true || table.guest_available === true;
}

export function isTrainerPatternGuestAvailable(tables = [], pattern = '') {
  const normalizedPattern = String(pattern || '');
  return tables.some((table) => (
    String(table?.pattern || '') === normalizedPattern
    && (table.guestAvailable === true || table.guest_available === true)
  ));
}

export function buildTrainerQueryPayloadForActor(payload, actor) {
  if (!isGuestTrainerActor(actor)) return payload;
  return {
    ...payload,
    prefetch_rng: null,
    allow_prefetch: false,
    guest_foreground_query: true,
  };
}

export function trainerGuestErrorNotice(code) {
  if (
    code === TRAINER_GUEST_QUERY_LIMIT_CODE
    || code === TRAINER_GUEST_QUERY_EXHAUSTED_CODE
    || code === TRAINER_GUEST_NETWORK_LIMIT_CODE
  ) return 'exhausted';
  if (
    code === TRAINER_GUEST_TABLE_LOCKED_CODE
    || code === TRAINER_GUEST_TABLE_LOGIN_REQUIRED_CODE
  ) return 'locked';
  return '';
}

export function createTrainerLookupAccessCoordinator({
  getActor,
  ensureGuestSession,
  resolveEnsuredActor = (value) => value,
  validateActor = () => true,
  onFailure = () => {},
} = {}) {
  const pending = new Map();
  let ensurePromise = null;

  const flush = (actor) => {
    const actions = [...pending.values()];
    pending.clear();
    for (const action of actions) {
      if (validateActor(actor, action.context) !== true) continue;
      action.callback(actor);
    }
  };

  const request = (key, callback, context = {}) => {
    const actor = getActor?.() || null;
    if (actor) {
      if (validateActor(actor, context) !== true) return false;
      callback(actor);
      return true;
    }
    if (typeof ensureGuestSession !== 'function') {
      onFailure(new Error('Guest session support is unavailable.'));
      return false;
    }
    pending.set(String(key), { callback, context });
    if (!ensurePromise) {
      ensurePromise = Promise.resolve()
        .then(() => ensureGuestSession())
        .then((result) => {
          const ensuredActor = resolveEnsuredActor(result);
          if (!ensuredActor) throw new Error('Guest session was not established.');
          flush(ensuredActor);
        })
        .catch((error) => {
          pending.clear();
          onFailure(error);
        })
        .finally(() => {
          ensurePromise = null;
        });
    }
    return true;
  };

  return {
    request,
    clear: () => pending.clear(),
    pendingCount: () => pending.size,
  };
}
