import { nextTick } from "vue";

export async function insertAtCursor(model, input, snippet) {
  const start = input?.selectionStart ?? model.value.length;
  const end = input?.selectionEnd ?? start;
  const before = model.value.slice(0, start),
    after = model.value.slice(end);
  const insert =
    (before && !before.endsWith("\n") ? "\n" : "") +
    snippet +
    (after.startsWith("\n") ? "" : "\n");
  model.value = before + insert + after;
  await nextTick();
  input?.focus();
  input?.setSelectionRange(start + insert.length, start + insert.length);
}
