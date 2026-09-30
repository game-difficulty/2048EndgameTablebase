// Document PiP moves the live surface to another visible document; the source
// tab becoming hidden must not freeze that surface (or the video PiP renderer).
export function shouldRefreshLiveClock(pipActive = false, doc = document, host = window) {
  const pipWindow = host.documentPictureInPicture?.window;
  return !doc.hidden || pipActive || !!doc.pictureInPictureElement
    || !!(pipWindow && !pipWindow.closed);
}
