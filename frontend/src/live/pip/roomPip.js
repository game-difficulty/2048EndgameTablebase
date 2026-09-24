// Room-level presentation infrastructure. It knows no game, lane or wire protocol.
export function choosePipMode(requested, { document: documentSupported, video, renderer }) {
  if (requested === 'document') return documentSupported ? 'document' : null;
  if (requested === 'video') return video && renderer ? 'video' : null;
  return documentSupported ? 'document' : video && renderer ? 'video' : null;
}

export function moveSurface(surface, destination) {
  const marker = surface.ownerDocument.createComment('room-pip-return');
  surface.before(marker);
  destination.append(surface);
  let restored = false;
  return () => {
    if (restored) return;
    restored = true;
    if (marker.parentNode) marker.replaceWith(surface);
  };
}

export function copyRoomStyles(source, target) {
  for (const sheet of source.styleSheets) {
    const style = target.createElement('style');
    try {
      style.textContent = [...sheet.cssRules].map(rule => rule.cssText).join('\n');
      target.head.append(style);
    } catch {
      if (sheet.href) {
        const link = target.createElement('link');
        link.rel = 'stylesheet'; link.href = sheet.href; target.head.append(link);
      }
    }
  }
  const base = target.createElement('base'); base.href = source.baseURI; target.head.prepend(base);
}

// getFrame => { key, width, height, draw(ctx) }. Only the content adapter understands
// its logical data. Frames are generated locally; no screen capture or video upload.
export class FramePump {
  constructor(getFrame, draw, now = () => performance.now()) {
    this.getFrame = getFrame; this.draw = draw; this.now = now;
    this.lastKey = null; this.lastAt = -Infinity; this.draws = 0; this.updates = 0;
  }
  tick(force = false) {
    const at = this.now();
    if (!force && at - this.lastAt < 100) return false;
    const frame = this.getFrame();
    if (!frame) throw Error('content_has_no_video_pip_adapter');
    if (!force && frame.key === this.lastKey && at - this.lastAt < 1000) return false;
    this.draw(frame);
    if (frame.key !== this.lastKey) this.updates++;
    this.lastKey = frame.key; this.lastAt = at; this.draws++;
    return true;
  }
}
