(function initialiseReplayViewer() {
  'use strict';

  const {
    ReplayFormatError,
    UNKNOWN_TIMING_FALLBACK_MS,
    decodeReplayBytes,
    decodeReplayText,
    playbackDuration,
    playbackTimelineAtState,
    planMoveTransitions,
    progressAtPlaybackTimeline,
    replayTimeAtPlaybackTimeline,
    snapshotToHex,
  } = window.ReplayCore;

  const elements = {
    app: document.querySelector('#app'),
    board: document.querySelector('#board'),
    boardHexValue: document.querySelector('#board-hex-value'),
    boardPlaceholder: document.querySelector('#board-placeholder'),
    dropHint: document.querySelector('#drop-hint'),
    error: document.querySelector('#error-message'),
    fileInput: document.querySelector('#file-input'),
    fileName: document.querySelector('#file-name'),
    inputButton: document.querySelector('#input-button'),
    inputCancel: document.querySelector('#input-cancel'),
    inputCode: document.querySelector('#input-code'),
    inputDialog: document.querySelector('#input-dialog'),
    inputLoad: document.querySelector('#input-load'),
    uploadButton: document.querySelector('#upload-button'),
    speedButton: document.querySelector('#speed-button'),
    speedCancel: document.querySelector('#speed-cancel'),
    speedDialog: document.querySelector('#speed-dialog'),
    speedSave: document.querySelector('#speed-save'),
    originalRate: document.querySelector('#original-rate'),
    constantMs: document.querySelector('#constant-ms'),
    fixedSpeedFields: document.querySelector('#fixed-speed-fields'),
    originalSpeedFields: document.querySelector('#original-speed-fields'),
    score: document.querySelector('#score-value'),
    nextTime: document.querySelector('#next-time-value'),
    totalTime: document.querySelector('#total-time-value'),
    step: document.querySelector('#step-value'),
    timingNote: document.querySelector('#timing-note'),
    timeline: document.querySelector('#node-timeline'),
    progress: document.querySelector('#progress'),
    progressFill: document.querySelector('#progress-fill'),
    progressCurrent: document.querySelector('#progress-current'),
    progressTotal: document.querySelector('#progress-total'),
    backTen: document.querySelector('#back-ten'),
    backOne: document.querySelector('#back-one'),
    playPause: document.querySelector('#play-pause'),
    forwardOne: document.querySelector('#forward-one'),
    forwardTen: document.querySelector('#forward-ten'),
  };

  const state = {
    replay: null,
    progress: 0,
    playing: false,
    playbackMode: 'original',
    originalRate: 1,
    constantMs: 100,
    animationFrame: null,
    clockMs: 0,
    anchorWallMs: 0,
    anchorPlaybackTimelineMs: 0,
    tileElements: [],
    motionLayer: null,
    boardAnimationTimer: null,
    milestoneTimeElements: [],
  };

  function showError(message) {
    elements.error.textContent = message || '';
    elements.error.hidden = !message;
  }

  function formatClock(milliseconds) {
    const safe = Math.max(0, Math.round(milliseconds));
    const hours = Math.floor(safe / 3600000);
    const minutes = Math.floor((safe % 3600000) / 60000);
    const seconds = Math.floor((safe % 60000) / 1000);
    const millis = safe % 1000;
    if (hours > 0) {
      return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
    }
    return `${minutes}:${String(seconds).padStart(2, '0')}.${String(millis).padStart(3, '0')}`;
  }

  function formatNodeTime(milliseconds) {
    if (milliseconds < 60000) return (milliseconds / 1000).toFixed(3);
    return formatClock(milliseconds);
  }

  function formatStepTime(step) {
    if (!step) return '—';
    if (step.deltaMs === null) return `未知 · ${UNKNOWN_TIMING_FALLBACK_MS} ms`;
    if (step.deltaMs < 1000) return `${step.deltaMs} ms`;
    return `${(step.deltaMs / 1000).toFixed(3)} s`;
  }

  function tileClass(value) {
    if (value <= 65536) return `value-${value}`;
    return 'value-super';
  }

  function setTileAppearance(tile, exponent, extraClass = '') {
    const value = exponent ? 2 ** exponent : 0;
    tile.className = value
      ? `tile ${tileClass(value)}${extraClass}`
      : `tile is-empty${extraClass}`;
    tile.firstElementChild.textContent = value || '';
  }

  function clearBoardAnimation() {
    if (state.boardAnimationTimer !== null) {
      window.clearTimeout(state.boardAnimationTimer);
      state.boardAnimationTimer = null;
    }
    if (state.motionLayer) state.motionLayer.replaceChildren();
    for (const tile of state.tileElements) tile.classList.remove('is-animation-hidden');
  }

  function createBoard() {
    const replay = state.replay;
    elements.board.innerHTML = '';
    state.tileElements = [];
    elements.board.style.setProperty('--board-width', replay.width);
    elements.board.style.setProperty('--board-height', replay.height);
    elements.board.style.aspectRatio = `${replay.width} / ${replay.height}`;

    for (let index = 0; index < replay.cellCount; index += 1) {
      const cell = document.createElement('div');
      cell.className = 'board-cell';
      const tile = document.createElement('div');
      tile.className = 'tile is-empty';
      const text = document.createElement('span');
      tile.append(text);
      cell.append(tile);
      elements.board.append(cell);
      state.tileElements.push(tile);
    }
    state.motionLayer = document.createElement('div');
    state.motionLayer.className = 'tile-motion-layer';
    state.motionLayer.setAttribute('aria-hidden', 'true');
    elements.board.append(state.motionLayer);
    elements.board.hidden = false;
    elements.boardPlaceholder.hidden = true;
  }

  function createTimeline() {
    elements.timeline.innerHTML = '';
    state.milestoneTimeElements = [];
    for (const milestone of state.replay.milestones) {
      const row = document.createElement('div');
      row.className = 'node-row';
      const badge = document.createElement('div');
      badge.className = `node-badge ${tileClass(Number(milestone.label))}`;
      badge.textContent = milestone.label;
      badge.title = `${milestone.key} 节点`;
      const time = document.createElement('div');
      time.className = 'node-time';
      time.textContent = '';
      row.append(badge, time);
      elements.timeline.append(row);
      state.milestoneTimeElements.push(time);
    }
  }

  function createMotionTile(exponent, positionIndex, extraClass = '') {
    const cell = state.tileElements[positionIndex].parentElement;
    const tile = document.createElement('div');
    const text = document.createElement('span');
    tile.append(text);
    setTileAppearance(tile, exponent, ` motion-tile${extraClass}`);
    tile.style.left = `${cell.offsetLeft}px`;
    tile.style.top = `${cell.offsetTop}px`;
    tile.style.width = `${cell.offsetWidth}px`;
    tile.style.height = `${cell.offsetHeight}px`;
    state.motionLayer.append(tile);
    return tile;
  }

  function renderBoard(fromProgress = null) {
    if (!state.replay) return;
    clearBoardAnimation();
    const board = state.replay.getBoardAt(state.progress);
    const shouldAnimate =
      fromProgress !== null &&
      state.progress === fromProgress + 1 &&
      !window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

    for (let index = 0; index < board.length; index += 1) {
      setTileAppearance(
        state.tileElements[index],
        board[index],
        shouldAnimate ? ' is-animation-hidden' : '',
      );
    }

    if (!shouldAnimate) return;

    const step = state.replay.steps[fromProgress];
    const before = state.replay.getBoardAt(fromProgress);
    const transition = planMoveTransitions(
      before,
      state.replay.width,
      state.replay.height,
      step.direction,
      step.number,
      step.special32k,
    );
    const movingTiles = transition.sources.map((source) => {
      const tile = createMotionTile(source.exponent, source.fromIndex, ' is-moving');
      const fromCell = state.tileElements[source.fromIndex].parentElement;
      const toCell = state.tileElements[source.toIndex].parentElement;
      return {
        tile,
        translateX: toCell.offsetLeft - fromCell.offsetLeft,
        translateY: toCell.offsetTop - fromCell.offsetTop,
      };
    });

    for (const merge of transition.merges) {
      createMotionTile(merge.exponent, merge.toIndex, ' is-merge-result');
    }
    const spawnIndex = step.spawnY * state.replay.width + step.spawnX;
    createMotionTile(board[spawnIndex], spawnIndex, ' is-new-result');

    // Force the source positions to commit before applying destination
    // transforms. The original site uses the same 100 ms slide followed by
    // 200 ms merge/spawn effects.
    void state.motionLayer.offsetWidth;
    for (const movement of movingTiles) {
      movement.tile.style.transform =
        `translate(${movement.translateX}px, ${movement.translateY}px)`;
    }

    state.boardAnimationTimer = window.setTimeout(() => {
      state.boardAnimationTimer = null;
      state.motionLayer.replaceChildren();
      for (const tile of state.tileElements) tile.classList.remove('is-animation-hidden');
    }, 310);
  }

  function renderTimeline() {
    if (!state.replay) return;
    state.replay.milestones.forEach((milestone, index) => {
      const timeElement = state.milestoneTimeElements[index];
      const reached = milestone.reachedStep !== null && state.progress >= milestone.reachedStep;
      timeElement.textContent = reached
        ? formatNodeTime(milestone.timeMs)
        : '';
      timeElement.classList.toggle('is-reached', reached);
    });
  }

  function renderControls() {
    const loaded = Boolean(state.replay);
    for (const control of [
      elements.backTen,
      elements.backOne,
      elements.playPause,
      elements.forwardOne,
      elements.forwardTen,
      elements.progress,
    ]) {
      control.disabled = !loaded;
    }
    elements.playPause.textContent = state.playing ? '❚❚' : '▶';
    elements.playPause.setAttribute('aria-label', state.playing ? '暂停' : '播放');
    elements.playPause.title = state.playing ? '暂停（空格）' : '播放（空格）';
  }

  function renderElapsedTime(replayTimeMs = state.clockMs) {
    if (!state.replay) {
      elements.totalTime.textContent = '0:00.000';
      return;
    }
    elements.totalTime.textContent = formatClock(replayTimeMs);
  }

  function renderStats(replayTimeMs = state.clockMs) {
    if (!state.replay) {
      elements.score.textContent = '0';
      elements.nextTime.textContent = '—';
      elements.boardHexValue.value = '';
      renderElapsedTime(0);
      elements.step.textContent = '0 / 0';
      elements.progressCurrent.textContent = '0';
      elements.progressTotal.textContent = '0';
      elements.progressFill.style.width = '0%';
      return;
    }

    const replay = state.replay;
    const nextStep = state.progress < replay.moveCount ? replay.steps[state.progress] : null;
    const boardHex = snapshotToHex(replay.getBoardAt(state.progress));
    if (elements.boardHexValue.value !== boardHex) {
      elements.boardHexValue.value = boardHex;
    }
    elements.score.textContent = Math.round(replay.scores[state.progress]).toLocaleString('zh-CN');
    elements.nextTime.textContent = formatStepTime(nextStep);
    renderElapsedTime(replayTimeMs);
    elements.step.textContent = `${state.progress.toLocaleString('zh-CN')} / ${replay.moveCount.toLocaleString('zh-CN')}`;
    elements.progress.value = String(state.progress);
    elements.progressCurrent.textContent = state.progress.toLocaleString('zh-CN');
    elements.progressTotal.textContent = replay.moveCount.toLocaleString('zh-CN');
    const progressPercent = replay.moveCount
      ? (state.progress / replay.moveCount) * 100
      : 0;
    elements.progressFill.style.width = `${progressPercent}%`;
  }

  function renderAll(replayTimeMs = state.clockMs, boardFromProgress = null) {
    renderBoard(boardFromProgress);
    renderTimeline();
    renderStats(replayTimeMs);
    renderControls();
  }

  function selectBoardHex() {
    elements.boardHexValue.select();
  }

  function playbackRate() {
    return state.playbackMode === 'original' ? state.originalRate : 1;
  }

  function currentPlaybackTimeline(now = performance.now()) {
    if (!state.replay) return 0;
    const timeline = state.playing
      ? state.anchorPlaybackTimelineMs + (now - state.anchorWallMs) * playbackRate()
      : playbackTimelineAtState(
        state.replay,
        state.progress,
        state.clockMs,
        state.playbackMode,
        state.constantMs,
      );
    return Math.max(
      0,
      Math.min(
        playbackDuration(state.replay, state.playbackMode, state.constantMs),
        timeline,
      ),
    );
  }

  function pause(shouldRender = true) {
    if (state.playing && state.replay) {
      const timeline = currentPlaybackTimeline();
      state.progress = progressAtPlaybackTimeline(
        state.replay,
        timeline,
        state.playbackMode,
        state.constantMs,
      );
      state.clockMs = replayTimeAtPlaybackTimeline(
        state.replay,
        timeline,
        state.playbackMode,
        state.constantMs,
      );
    }
    state.playing = false;
    if (state.animationFrame !== null) window.cancelAnimationFrame(state.animationFrame);
    state.animationFrame = null;
    if (shouldRender) renderAll();
  }

  function setProgress(progress, shouldPause = true, animateForward = false) {
    if (!state.replay) return;
    const previousProgress = state.progress;
    if (shouldPause) pause(false);
    state.progress = Math.max(0, Math.min(state.replay.moveCount, Math.round(progress)));
    state.clockMs = state.replay.cumulativeMs[state.progress];
    renderAll(state.clockMs, animateForward ? previousProgress : null);
  }

  function playbackTick(now) {
    if (!state.playing || !state.replay) return;
    const timeline = currentPlaybackTimeline(now);
    const dueProgress = progressAtPlaybackTimeline(
      state.replay,
      timeline,
      state.playbackMode,
      state.constantMs,
    );
    const replayTimeMs = replayTimeAtPlaybackTimeline(
      state.replay,
      timeline,
      state.playbackMode,
      state.constantMs,
    );
    const previousProgress = state.progress;
    const progressChanged = dueProgress !== previousProgress;
    state.progress = dueProgress;
    state.clockMs = replayTimeMs;

    if (timeline >= playbackDuration(state.replay, state.playbackMode, state.constantMs)) {
      state.progress = state.replay.moveCount;
      state.clockMs = state.replay.playbackTimeMs;
      state.playing = false;
      state.animationFrame = null;
      renderAll();
      return;
    }

    if (progressChanged) renderAll(replayTimeMs, previousProgress);
    else renderElapsedTime(replayTimeMs);

    // Every frame derives from one absolute wall-clock anchor, so rendering
    // delays can cause catch-up but can never accumulate into the stopwatch.
    state.animationFrame = window.requestAnimationFrame(playbackTick);
  }

  function play() {
    if (!state.replay) return;
    if (state.playing) {
      pause();
      return;
    }
    if (state.progress >= state.replay.moveCount) setProgress(0, false);
    state.anchorPlaybackTimelineMs = playbackTimelineAtState(
      state.replay,
      state.progress,
      state.clockMs,
      state.playbackMode,
      state.constantMs,
    );
    state.playing = true;
    state.anchorWallMs = performance.now();
    renderControls();
    state.animationFrame = window.requestAnimationFrame(playbackTick);
  }

  function renderSpeedButton() {
    elements.speedButton.textContent = state.playbackMode === 'original'
      ? `原始步速 · ${state.originalRate}×`
      : `恒定步速 · ${state.constantMs} ms`;
  }

  function updateSpeedFieldVisibility() {
    const selected = document.querySelector('input[name="speed-mode"]:checked').value;
    elements.originalSpeedFields.hidden = selected !== 'original';
    elements.fixedSpeedFields.hidden = selected !== 'constant';
  }

  async function installReplay(loader, sourceName) {
    pause(false);
    clearBoardAnimation();
    showError('');
    elements.fileName.textContent = '正在解析回放…';
    await new Promise((resolve) => window.requestAnimationFrame(resolve));

    try {
      const replay = loader();
      state.replay = replay;
      state.progress = 0;
      state.clockMs = 0;
      elements.progress.min = '0';
      elements.progress.max = String(replay.moveCount);
      elements.fileName.textContent = `${sourceName} · ${replay.width}×${replay.height} · ${replay.moveCount.toLocaleString('zh-CN')} 步`;
      elements.timingNote.textContent = replay.unknownTimings
        ? `含 ${replay.unknownTimings} 个未知间隔；统一按 ${UNKNOWN_TIMING_FALLBACK_MS} ms 计入回放用时。秒表采用绝对时间基准，不累计页面渲染延迟。`
        : '秒表采用绝对时间基准连续计时，不累计页面渲染延迟。';
      createBoard();
      createTimeline();
      renderAll();
    } catch (error) {
      state.replay = null;
      state.progress = 0;
      state.clockMs = 0;
      state.tileElements = [];
      state.motionLayer = null;
      state.milestoneTimeElements = [];
      elements.fileName.textContent = '尚未载入回放';
      elements.board.hidden = true;
      elements.boardPlaceholder.hidden = false;
      elements.timeline.innerHTML = '<p class="timeline-empty">载入回放后显示</p>';
      elements.timingNote.textContent = '';
      elements.progress.min = '0';
      elements.progress.max = '0';
      elements.progress.value = '0';
      const prefix = error instanceof ReplayFormatError ? '回放格式错误' : '无法读取回放';
      showError(`${prefix}：${error.message}`);
      renderStats();
      renderControls();
      throw error;
    }
  }

  async function loadFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.vrs')) {
      showError('请选择扩展名为 .vrs 的回放文件。');
      return;
    }
    const buffer = await file.arrayBuffer();
    try {
      await installReplay(() => decodeReplayBytes(buffer), file.name);
    } catch (_) {
      // The inline error already contains the actionable format detail.
    }
  }

  async function loadRankedReplayFromUrl() {
    const liveId = new URLSearchParams(window.location.search).get('live');
    if (liveId) {
      elements.fileName.textContent = '正在载入 AI 直播回放…';
      try {
        const response = await fetch(`/api/live/replays/${encodeURIComponent(liveId)}`);
        if (!response.ok) throw new Error(response.status === 404 ? '回放已过期或不存在' : '暂时无法获取回放');
        const responseText = await response.text();
        await installReplay(() => decodeReplayText(responseText), 'AI 直播对局');
      } catch (error) {
        showError(`无法载入直播回放：${error?.message || '网络请求失败'}`);
        elements.fileName.textContent = '直播回放载入失败';
      }
      return;
    }
    const replayId = new URLSearchParams(window.location.search).get('ranked');
    if (!replayId) return;
    elements.fileName.textContent = '正在载入已验证对局…';
    try {
      const response = await fetch(`/api/gamer/replays/${encodeURIComponent(replayId)}`, {
        headers: { Accept: 'application/json' },
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || !payload?.record_encoding) {
        throw new Error(payload?.detail || `HTTP ${response.status}`);
      }
      const source = `${payload.display_name || '排行榜对局'} · ${Number(payload.score || 0).toLocaleString('zh-CN')} 分`;
      await installReplay(() => decodeReplayText(payload.record_encoding), source);
    } catch (error) {
      showError(`无法载入排行榜对局：${error?.message || '网络请求失败'}`);
      elements.fileName.textContent = '排行榜对局载入失败';
    }
  }

  function openDialog(dialog) {
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
  }

  function closeDialog(dialog) {
    if (typeof dialog.close === 'function') dialog.close();
    else dialog.removeAttribute('open');
  }

  elements.inputButton.addEventListener('click', () => {
    showError('');
    openDialog(elements.inputDialog);
    window.setTimeout(() => elements.inputCode.focus(), 0);
  });
  elements.inputCancel.addEventListener('click', () => closeDialog(elements.inputDialog));
  elements.inputLoad.addEventListener('click', async () => {
    const text = elements.inputCode.value;
    if (!text.trim()) return;
    closeDialog(elements.inputDialog);
    try {
      await installReplay(() => decodeReplayText(text), '粘贴的回放代码');
    } catch (_) {
      // The inline error already contains the actionable format detail.
    }
  });

  elements.uploadButton.addEventListener('click', () => {
    // Selecting the same replay twice does not fire `change` unless the
    // previous file value is cleared first (for example after loading pasted
    // code between the two uploads).
    elements.fileInput.value = '';
    elements.fileInput.click();
  });
  elements.fileInput.addEventListener('change', () => loadFile(elements.fileInput.files[0]));

  elements.speedButton.addEventListener('click', () => {
    const selected = document.querySelector(`input[name="speed-mode"][value="${state.playbackMode}"]`);
    selected.checked = true;
    elements.originalRate.value = String(state.originalRate);
    elements.constantMs.value = String(state.constantMs);
    updateSpeedFieldVisibility();
    openDialog(elements.speedDialog);
  });
  elements.speedCancel.addEventListener('click', () => closeDialog(elements.speedDialog));
  document.querySelectorAll('input[name="speed-mode"]').forEach((radio) => {
    radio.addEventListener('change', updateSpeedFieldVisibility);
  });
  elements.speedSave.addEventListener('click', () => {
    const mode = document.querySelector('input[name="speed-mode"]:checked').value;
    const rate = Number.parseFloat(elements.originalRate.value);
    const constant = Number.parseFloat(elements.constantMs.value);
    if (mode === 'original' && (!Number.isFinite(rate) || rate <= 0)) {
      elements.originalRate.focus();
      return;
    }
    if (mode === 'constant' && (!Number.isFinite(constant) || constant < 0)) {
      elements.constantMs.focus();
      return;
    }
    pause(false);
    state.playbackMode = mode;
    state.originalRate = rate;
    state.constantMs = constant;
    renderSpeedButton();
    closeDialog(elements.speedDialog);
    renderAll();
  });

  elements.boardHexValue.addEventListener('focus', selectBoardHex);
  elements.boardHexValue.addEventListener('click', selectBoardHex);

  elements.backTen.addEventListener('click', () => setProgress(state.progress - 10));
  elements.backOne.addEventListener('click', () => setProgress(state.progress - 1));
  elements.playPause.addEventListener('click', play);
  elements.forwardOne.addEventListener('click', () => setProgress(state.progress + 1, true, true));
  elements.forwardTen.addEventListener('click', () => setProgress(state.progress + 10));
  elements.progress.addEventListener('input', () => setProgress(Number(elements.progress.value)));

  document.addEventListener('keydown', (event) => {
    if (event.target.closest('textarea, dialog, input:not([readonly])')) return;
    if (event.code === 'Space') {
      event.preventDefault();
      play();
    } else if (event.key === 'Enter') {
      event.preventDefault();
      setProgress(state.progress + 1, true, true);
    } else if (event.key === 'Backspace') {
      event.preventDefault();
      setProgress(state.progress - 1);
    } else if (event.key === 'ArrowLeft') {
      event.preventDefault();
      setProgress(state.progress - (event.shiftKey ? 10 : 1));
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      setProgress(
        state.progress + (event.shiftKey ? 10 : 1),
        true,
        !event.shiftKey,
      );
    }
  });

  for (const eventName of ['dragenter', 'dragover']) {
    document.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropHint.hidden = false;
    });
  }
  document.addEventListener('dragleave', (event) => {
    if (!event.relatedTarget) elements.dropHint.hidden = true;
  });
  document.addEventListener('drop', (event) => {
    event.preventDefault();
    elements.dropHint.hidden = true;
    loadFile(event.dataTransfer.files[0]);
  });

  renderSpeedButton();
  renderStats();
  renderControls();
  loadRankedReplayFromUrl();
})();
