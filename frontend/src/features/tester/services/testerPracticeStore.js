const STORAGE_KEY = '2048tables:tester-practice:v1';

const storageOrNull = () => {
  try {
    return typeof window !== 'undefined' ? window.sessionStorage : null;
  } catch (_error) {
    return null;
  }
};

const replacer = (_key, value) => (
  typeof value === 'bigint' ? { __testerBigInt: value.toString(16) } : value
);

const reviver = (_key, value) => (
  value
  && typeof value === 'object'
  && typeof value.__testerBigInt === 'string'
    ? BigInt(`0x${value.__testerBigInt || '0'}`)
    : value
);

const validPractice = (record) => (
  record?.version === 1
  && Number.isInteger(Number(record.userId))
  && Array.isArray(record?.session?.practice?.board)
  && record.session.practice.board.length === 16
  && /^[0-9a-f]{16}$/iu.test(String(record.session.practice.boardHex || ''))
  && Array.isArray(record?.session?.records)
);

export function saveTesterPracticeState(payload) {
  const storage = storageOrNull();
  if (!storage) return false;
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify({
      ...payload,
      version: 1,
      savedAt: Date.now(),
    }, replacer));
    return true;
  } catch (_error) {
    return false;
  }
}

export function restoreTesterPracticeState() {
  const storage = storageOrNull();
  if (!storage) return null;
  try {
    const record = JSON.parse(storage.getItem(STORAGE_KEY) || 'null', reviver);
    return validPractice(record) ? record : null;
  } catch (_error) {
    return null;
  }
}

export function clearTesterPracticeState() {
  try {
    storageOrNull()?.removeItem(STORAGE_KEY);
  } catch (_error) {
    // Storage is an optional recovery layer.
  }
}
