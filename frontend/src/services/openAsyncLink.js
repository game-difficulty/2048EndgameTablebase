// Reserve the tab during the click gesture, before waiting for the server.
export async function openAsyncLink(loadUrl, browser = window) {
  const tab = browser.open('about:blank', '_blank');
  if (tab) tab.opener = null;
  try {
    const url = await loadUrl();
    if (tab) {
      if (!tab.closed) tab.location.replace(url);
    } else {
      browser.location.assign(url);
    }
  } catch (error) {
    if (tab && !tab.closed) tab.close();
    throw error;
  }
}
