const STORAGE_KEY = '2048tables:trainer-practice:v1';

const storageOrNull = () => {
  try {
    return typeof window !== 'undefined' ? window.sessionStorage : null;
  } catch (_error) {
    return null;
  }
};

const validSnapshot = (record) => (
  record?.version === 1
  && Number.isInteger(Number(record.userId))
  && Array.isArray(record?.practice?.board)
  && record.practice.board.length === 16
  && /^[0-9a-f]{16}$/iu.test(String(record.practice.boardHex || ''))
  && Array.isArray(record.practice.history)
);

export function saveTrainerPracticeState(payload) {
  const storage = storageOrNull();
  if (!storage) return false;
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify({
      ...payload,
      version: 1,
      savedAt: Date.now(),
    }));
    return true;
  } catch (_error) {
    return false;
  }
}

export function restoreTrainerPracticeState() {
  const storage = storageOrNull();
  if (!storage) return null;
  try {
    const record = JSON.parse(storage.getItem(STORAGE_KEY) || 'null');
    return validSnapshot(record) ? record : null;
  } catch (_error) {
    return null;
  }
}

export function clearTrainerPracticeState() {
  try {
    storageOrNull()?.removeItem(STORAGE_KEY);
  } catch (_error) {
    // Session recovery is optional.
  }
}
