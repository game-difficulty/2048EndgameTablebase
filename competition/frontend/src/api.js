const DEVICE_SESSION_KEY = '2048tables:device-session-token';
const apiOrigin = String(import.meta.env.VITE_COMPETITION_API_ORIGIN || '').replace(/\/$/, '');
const devUser = String(import.meta.env.VITE_COMPETITION_DEV_USER || '').trim();

function storedToken() {
  for (const storage of [window.localStorage, window.sessionStorage]) {
    try {
      const payload = JSON.parse(storage.getItem(DEVICE_SESSION_KEY) || 'null');
      if (payload?.token) return String(payload.token);
    } catch (_error) {
      // Continue with cookies or the development identity.
    }
  }
  return '';
}

function headers(hasBody = false) {
  const result = hasBody ? { 'Content-Type': 'application/json' } : {};
  const token = storedToken();
  if (token) result.Authorization = `Bearer ${token}`;
  if (devUser) result['X-Competition-Dev-User'] = devUser;
  return result;
}

async function request(path, { method = 'GET', body, timeoutMs } = {}) {
  const controller = timeoutMs ? new AbortController() : null;
  const timeout = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;
  try {
    const response = await fetch(`${apiOrigin}${path}`, {
      method,
      headers: headers(body !== undefined),
      credentials: 'include',
      body: body === undefined ? undefined : JSON.stringify(body),
      ...(controller ? { signal: controller.signal } : {}),
    });
    // A truncated/aborted success body is not an acknowledgement: the state
    // sender must retry, particularly when this was the final result.
    const payload = response.ok ? await response.json() : await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = payload?.detail;
      const error = new Error(
        (typeof detail === 'object' ? detail?.message : detail) || `请求失败（${response.status}）`,
      );
      error.code = typeof detail === 'object' ? detail?.code : '';
      error.status = response.status;
      throw error;
    }
    return payload;
  } finally { if (timeout !== null) clearTimeout(timeout); }
}

export function commandId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `cmd-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export const api = {
  session: (options) => request('/api/session', options),
  practiceAppearance: () => request('/api/practice/appearance'),
  practiceLeaderboard: (projectId) => request(`/api/practice/${encodeURIComponent(projectId)}/leaderboard`),
  submitPracticeResult: (projectId, result) => request(`/api/practice/${encodeURIComponent(projectId)}/results`, {
    method: 'POST', body: result,
  }),
  list: () => request('/api/competitions'),
  events: () => request('/api/events'),
  event: (slug) => request(`/api/events/${encodeURIComponent(slug)}`),
  eventStatistics: (slug) => request(`/api/events/${encodeURIComponent(slug)}/statistics`),
  enrollment: (slug) => request(`/api/events/${encodeURIComponent(slug)}/enrollment`),
  enrollmentAction: (slug, body) => request(`/api/events/${encodeURIComponent(slug)}/enrollment/actions`, { method: 'POST', body }),
  importEnrollment: (slug, body) => request(`/api/events/${encodeURIComponent(slug)}/enrollment/import`, { method: 'POST', body }),
  importEventRoster: (slug, body) => request(`/api/events/${encodeURIComponent(slug)}/roster`, { method: 'POST', body }),
  createEvent: (body) => request('/api/events', { method: 'POST', body }),
  updateEvent: (slug, body) => request(`/api/events/${encodeURIComponent(slug)}/settings`, { method: 'POST', body }),
  linkEventRoom: (slug, roomCode) => request(`/api/events/${encodeURIComponent(slug)}/rooms`, {
    method: 'POST', body: { room_code: roomCode },
  }),
  create: (name, projects, eventSlug = null, schedule = {}) => request('/api/competitions', {
    method: 'POST',
    body: { name, projects, event_slug: eventSlug, ...schedule },
  }),
  room: (code) => request(`/api/competitions/${encodeURIComponent(code)}`),
  rematch: (code) => request(`/api/competitions/${encodeURIComponent(code)}/rematch`, {method:'POST',body:{command_id:commandId()}}),
  checkIn: (code) => request(`/api/competitions/${encodeURIComponent(code)}/check-in`, { method: 'POST' }),
  closeRoom: (code) => request(`/api/competitions/${encodeURIComponent(code)}/close`, {
    method: 'POST', body: { command_id: commandId() },
  }),
  claimSeat: (code, side, position) => request(
    `/api/competitions/${encodeURIComponent(code)}/seat`,
    { method: 'POST', body: { side, position, command_id: commandId() } },
  ),
  leaveSeat: (code) => request(
    `/api/competitions/${encodeURIComponent(code)}/seat/leave`,
    { method: 'POST', body: { command_id: commandId() } },
  ),
  manageMember: (code, userId, remove = true) => request(`/api/competitions/${encodeURIComponent(code)}/members/manage`, {
    method: 'POST', body: { user_id: userId, remove, command_id: commandId() },
  }),
  ready: (code, ready) => request(
    `/api/competitions/${encodeURIComponent(code)}/ready`,
    { method: 'POST', body: { ready, command_id: commandId() } },
  ),
  submitPickBan: (code, pickProjectKey, banProjectKey, phaseToken) => request(
    `/api/competitions/${encodeURIComponent(code)}/draft/pick-ban`,
    {
      method: 'POST',
      body: {
        pick_project_key: pickProjectKey,
        ban_project_key: banProjectKey,
        phase_token: phaseToken,
        command_id: commandId(),
      },
    },
  ),
  submitBlind: (code, projectKey, phaseToken) => request(
    `/api/competitions/${encodeURIComponent(code)}/draft/blind`,
    {
      method: 'POST',
      body: {
        project_key: projectKey,
        phase_token: phaseToken,
        command_id: commandId(),
      },
    },
  ),
  submitLineup: (code, assignments, phaseToken) => request(
    `/api/competitions/${encodeURIComponent(code)}/lineup`,
    {
      method: 'POST',
      body: {
        assignments,
        phase_token: phaseToken,
        command_id: commandId(),
      },
    },
  ),
  setGameReadiness: (code, readinessRole, ready, phaseToken) => request(
    `/api/competitions/${encodeURIComponent(code)}/games/current/readiness`,
    {
      method: 'POST',
      body: {
        readiness_role: readinessRole,
        ready,
        phase_token: phaseToken,
        command_id: commandId(),
      },
    },
  ),
  startCurrentGame: (code, phaseToken) => request(
    `/api/competitions/${encodeURIComponent(code)}/games/current/start`,
    { method: 'POST', body: { phase_token: phaseToken, command_id: commandId() } },
  ),
  syncClientGame: (code, packet, phaseToken) => request(
    `/api/competitions/${encodeURIComponent(code)}/games/current/state`,
    { method: 'POST', timeoutMs: 8000,
      body: { ...packet, elapsed_ms: Math.round(packet.elapsed_ms), phase_token: phaseToken } },
  ),
  confirmCurrentResult: (code, resultRevision, phaseToken) => request(
    `/api/competitions/${encodeURIComponent(code)}/games/current/result/confirm`,
    {
      method: 'POST',
      body: {
        result_revision: resultRevision,
        phase_token: phaseToken,
        command_id: commandId(),
      },
    },
  ),
  reportIssue: (code, category, details) => request(
    `/api/competitions/${encodeURIComponent(code)}/issues`,
    { method: 'POST', body: { category, details, command_id: commandId() } },
  ),
  resolveIssue: (code, issueId, status, resolutionNote) => request(
    `/api/competitions/${encodeURIComponent(code)}/issues/${encodeURIComponent(issueId)}/resolve`,
    {
      method: 'POST',
      body: { status, resolution_note: resolutionNote, command_id: commandId() },
    },
  ),
  suspendMatch: (code, reasonCode, reasonText, phaseToken) => request(
    `/api/competitions/${encodeURIComponent(code)}/suspension/start`,
    {
      method: 'POST',
      body: {
        reason_code: reasonCode,
        reason_text: reasonText,
        phase_token: phaseToken,
        command_id: commandId(),
      },
    },
  ),
  setSuspensionReadiness: (code, ready, phaseToken) => request(
    `/api/competitions/${encodeURIComponent(code)}/suspension/readiness`,
    {
      method: 'POST',
      body: { ready, phase_token: phaseToken, command_id: commandId() },
    },
  ),
  resumeMatch: (code, phaseToken) => request(
    `/api/competitions/${encodeURIComponent(code)}/suspension/resume`,
    { method: 'POST', body: { phase_token: phaseToken, command_id: commandId() } },
  ),
  overrideCurrentResult: (
    code, yellowScore, whiteScore, winnerSide, reason, resultRevision, phaseToken,
  ) => request(
    `/api/competitions/${encodeURIComponent(code)}/games/current/result/override`,
    {
      method: 'POST',
      body: {
        yellow_score: yellowScore,
        white_score: whiteScore,
        winner_side: winnerSide,
        reason,
        result_revision: resultRevision,
        phase_token: phaseToken,
        command_id: commandId(),
      },
    },
  ),
  forceAdvanceCurrentResult: (code, reason, phaseToken) => request(
    `/api/competitions/${encodeURIComponent(code)}/games/current/result/force-advance`,
    {
      method: 'POST',
      body: { reason, phase_token: phaseToken, command_id: commandId() },
    },
  ),
  forceFinishMatch: (code, winnerSide, reason, phaseToken) => request(
    `/api/competitions/${encodeURIComponent(code)}/force-finish`,
    {
      method: 'POST',
      body: {
        winner_side: winnerSide,
        reason,
        phase_token: phaseToken,
        command_id: commandId(),
      },
    },
  ),
};

export function connectRoom(code, handlers = {}) {
  let socket;
  let reconnectTimer;
  let closed = false;
  let failures = 0;
  const open = () => {
    const base = apiOrigin ? new URL(apiOrigin, window.location.href) : new URL(window.location.href);
    base.protocol = base.protocol === 'https:' ? 'wss:' : 'ws:';
    base.pathname = `/ws/rooms/${encodeURIComponent(code)}`;
    base.search = '';
    socket = new WebSocket(base.toString());
    socket.onopen = () => {
      socket.send(JSON.stringify({
        type: 'authenticate',
        data: { token: storedToken(), dev_user: devUser },
      }));
      handlers.onOpen?.();
    };
    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message?.type === 'room.snapshot') failures = 0;
        handlers.onMessage?.(message);
      } catch (error) {
        handlers.onError?.(error);
      }
    };
    socket.onerror = (event) => handlers.onError?.(event);
    socket.onclose = (event) => {
      handlers.onClose?.(event);
      if (!closed && ![4401, 4403, 4404, 1008].includes(event.code)) {
        reconnectTimer = window.setTimeout(open, Math.min(1200 * 2 ** failures++, 15000));
      }
    };
  };
  open();
  return () => {
    closed = true;
    window.clearTimeout(reconnectTimer);
    socket?.close();
  };
}
