// Never rotate the DOM: unsupported browsers retain their current layout.
export async function requestNativeLandscape(doc, orientation) {
  if (typeof orientation?.lock !== 'function') return false;
  let enteredFullscreen = false;
  try {
    if (!doc.fullscreenElement && typeof doc.documentElement?.requestFullscreen === 'function') {
      await doc.documentElement.requestFullscreen();
      enteredFullscreen = true;
    }
    await orientation.lock('landscape');
    return true;
  } catch {
    if (enteredFullscreen && doc.fullscreenElement) {
      try { await doc.exitFullscreen(); } catch { /* Browser may already have exited. */ }
    }
    return false;
  }
}
