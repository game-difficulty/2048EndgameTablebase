export const AI_PARTICIPANTS = Object.freeze(['Lume', 'Clari', 'Vero'].map((name, lane) => Object.freeze({ id: name.toLowerCase(), name, lane })));
export const participantName = slot => slot?.name || AI_PARTICIPANTS[slot?.lane ?? 0]?.name || '';
export const showEndNotice = (run, now) => Boolean(run?.ended_at && now < run.ended_at * 1000 + 5000);
