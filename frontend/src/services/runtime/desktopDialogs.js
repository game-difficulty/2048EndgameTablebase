const EMPTY_LIST = Object.freeze([]);

const DIALOG_METHOD_FALLBACKS = Object.freeze({
  select_folder: 'select_folder',
  select_open_record: 'select_open_record',
  select_open_replay_file: 'select_open_replay_file',
  select_analysis_files: 'select_analysis_files',
  select_save_record: 'select_save_record',
  select_save_tester_log: 'select_save_tester_log',
  select_save_tester_replay: 'select_save_tester_replay',
});

function getPywebviewApi() {
  if (typeof window === 'undefined') {
    return null;
  }
  const api = window.pywebview?.api;
  return api && typeof api === 'object' ? api : null;
}

function normalizeSinglePath(value) {
  if (typeof value === 'string') {
    const normalized = value.trim();
    return normalized || null;
  }
  if (Array.isArray(value)) {
    return normalizeSinglePath(value[0]);
  }
  return null;
}

function normalizePathList(value) {
  const rawItems = Array.isArray(value) ? value : [value];
  const normalizedItems = rawItems
    .map((item) => normalizeSinglePath(item))
    .filter(Boolean);
  return normalizedItems.length ? normalizedItems : EMPTY_LIST;
}

async function invokeDesktopDialog(api, dialogId) {
  if (typeof api.show_dialog === 'function') {
    return api.show_dialog(dialogId);
  }

  const legacyMethodName = DIALOG_METHOD_FALLBACKS[dialogId];
  if (legacyMethodName && typeof api[legacyMethodName] === 'function') {
    return api[legacyMethodName]();
  }

  return undefined;
}

export async function tryDesktopDialog(
  dialogId,
  { multiple = false } = {}
) {
  const api = getPywebviewApi();
  if (!api) {
    return { handled: false, value: multiple ? EMPTY_LIST : null };
  }

  try {
    const rawValue = await invokeDesktopDialog(api, dialogId);
    if (rawValue === undefined) {
      return { handled: false, value: multiple ? EMPTY_LIST : null };
    }
    return {
      handled: true,
      value: multiple ? normalizePathList(rawValue) : normalizeSinglePath(rawValue),
    };
  } catch (error) {
    console.error(`Desktop dialog '${dialogId}' failed:`, error);
    return { handled: false, value: multiple ? EMPTY_LIST : null };
  }
}
