const text = (value) => String(value ?? '').trim();

export function battleActorKind(candidate) {
  const explicit = text(candidate?.actor_kind || candidate?.kind).toLowerCase();
  if (explicit === 'guest' || explicit === 'user') return explicit;
  const key = text(candidate?.actor_key);
  if (key.startsWith('g:')) return 'guest';
  if (key.startsWith('u:')) return 'user';
  if (candidate?.guest_id != null) return 'guest';
  if (candidate?.user_id != null || candidate?.id != null) return 'user';
  return '';
}

export function battleActorKey(candidate) {
  const explicit = text(candidate?.actor_key);
  if (explicit) return explicit;
  const kind = battleActorKind(candidate);
  if (kind === 'guest') {
    const id = text(candidate?.guest_id ?? candidate?.id);
    return id ? `g:${id}` : '';
  }
  if (kind === 'user') {
    const id = text(candidate?.user_id ?? candidate?.id);
    return id ? `u:${id}` : '';
  }
  return '';
}

export function normalizeBattleActor(candidate) {
  const actorKey = battleActorKey(candidate);
  const kind = battleActorKind(candidate);
  if (!actorKey || !kind) return null;
  return {
    ...candidate,
    kind,
    actor_kind: kind,
    actor_key: actorKey,
    display_name: text(candidate?.display_name || candidate?.username || candidate?.email),
  };
}

export function sameBattleActor(left, right) {
  const leftKey = battleActorKey(left);
  const rightKey = battleActorKey(right);
  if (leftKey && rightKey) return leftKey === rightKey;
  const leftUser = left?.user_id ?? (battleActorKind(left) === 'user' ? left?.id : null);
  const rightUser = right?.user_id ?? (battleActorKind(right) === 'user' ? right?.id : null);
  return leftUser != null && rightUser != null && String(leftUser) === String(rightUser);
}

export function isBattleGuest(candidate) {
  return battleActorKind(candidate) === 'guest';
}

export function battleActorRenderKey(candidate, fallback = '') {
  return battleActorKey(candidate) || text(candidate?.user_id ?? candidate?.guest_id ?? fallback);
}

export function buildBattleKickPayload(actor, requestId) {
  const actorKey = battleActorKey(actor);
  const userId = typeof actor === 'object' ? actor?.user_id : actor;
  const body = { request_id: requestId };
  if (actorKey) body.actor_key = actorKey;
  if (battleActorKind(actor) !== 'guest' && userId != null && Number.isFinite(Number(userId))) {
    body.user_id = Number(userId);
  }
  return body;
}
