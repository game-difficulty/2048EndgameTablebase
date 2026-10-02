export function aspectRatio(style) {
  const parts = String(style.aspectRatio || '').replace(/^auto\s+/, '').split('/').map(Number);
  return parts[0] > 0 && (parts.length === 1 || parts[1] > 0)
    ? parts[0] / (parts[1] || 1) : 1;
}

export function chooseRuleSize(budget, measure) {
  const candidates = [
    { font: 24, padding: 10, lineHeight: 1.5 },
    { font: 24, padding: 4, lineHeight: 1.5 },
    ...Array.from({ length: 7 }, (_, index) => ({ font: 24 - index, padding: 4, lineHeight: 1.25 })),
  ];
  for (const candidate of candidates) {
    const height = measure(candidate);
    if (height <= budget || candidate.font === 18) return { ...candidate, height };
  }
}

export function fitBoard(width, height, ratio) {
  return Math.max(0, Math.min(width, 330, height * ratio));
}

// The renderer owns its intrinsic aspect ratio. No project IDs or shape lists
// belong here; measure the direct board root, including any entrance/exit area.
export function observeAdaptiveBoards(root) {
  let frame = 0, signature = '', areas = new Set();
  const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
  const resize = new ResizeObserver(schedule);
  resize.observe(root);
  function update() {
    frame = 0;
    const entries = [...root.querySelectorAll('.stream-board-area')].map(area => {
      const board = area.firstElementChild;
      if (!board) return null;
      if (!areas.has(area)) { resize.observe(area); areas.add(area); }
      const ratio = aspectRatio(getComputedStyle(board));
      return { area, board, ratio, width: area.clientWidth, height: area.clientHeight };
    }).filter(Boolean);
    for (const area of areas) if (!root.contains(area)) { resize.unobserve(area); areas.delete(area); }
    const layout = root.querySelector('.game-layout');
    const rule = layout?.querySelector('.game-rule-strip');
    const key = JSON.stringify([entries.map(e => [e.ratio, e.width, e.height]), layout?.clientHeight, rule?.clientWidth, rule?.textContent]);
    if (signature === key) return;
    signature = key;
    if (rule && entries.length) {
      const gridStyle = getComputedStyle(layout);
      const available = layout.clientHeight - parseFloat(gridStyle.paddingTop || 0) - parseFloat(gridStyle.paddingBottom || 0) - parseFloat(gridStyle.rowGap || 0);
      const needed = Math.max(...entries.map(e => {
        const pane = e.area.closest('.project-pane');
        return Math.min(e.width, 330) / e.ratio + (pane ? pane.getBoundingClientRect().height - e.height : 0);
      }));
      const probe = rule.cloneNode(true);
      Object.assign(probe.style, { position: 'fixed', left: '-10000px', top: '0', width: `${rule.clientWidth}px`, boxSizing: 'border-box', visibility: 'hidden', height: 'auto' });
      layout.append(probe);
      const result = chooseRuleSize(available - needed, ({ font, padding, lineHeight }) => {
        Object.assign(probe.style, { fontSize: `${font}px`, paddingTop: `${padding}px`, paddingBottom: `${padding}px`, lineHeight: String(lineHeight) });
        return probe.getBoundingClientRect().height;
      });
      probe.remove();
      rule.style.fontSize = `${result.font}px`;
      rule.style.paddingBlock = `${result.padding}px`;
      rule.style.lineHeight = String(result.lineHeight);
    }
    // Read after the rule allocation, then fit every board without distorting it.
    for (const { area, ratio } of entries) {
      const width = fitBoard(area.clientWidth, area.clientHeight, ratio);
      const value = `${width}px`;
      if (area.style.getPropertyValue('--stream-board-width') !== value) area.style.setProperty('--stream-board-width', value);
      area.classList.add('adaptive-board-fit');
    }
  }
  const mutation = new MutationObserver(schedule);
  mutation.observe(root, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['style', 'class'] });
  const fontsChanged = () => { signature = ''; schedule(); };
  document.fonts?.addEventListener('loadingdone', fontsChanged);
  schedule();
  return () => { resize.disconnect(); mutation.disconnect(); document.fonts?.removeEventListener('loadingdone', fontsChanged); cancelAnimationFrame(frame); };
}
