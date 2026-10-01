import { ref, shallowRef } from 'vue';
import * as engine from './engine.js';
import * as storage from './storage.js';
import { json, getStatus, upload } from './client.js';
import { needsReplayUpload } from './archivePolicy.js';
import { timerSplitsFor } from './timerSplits.js';
import { EventBuffer } from './eventBuffer.js';
import { reachedVictory } from './victory.js';

export const messages = {
  rollback_detected: '检测到本地进度落后于服务器记录，本局已判定回档，不能继续排位。',
  prefix_conflict: '本地记录与服务器已留档前缀不同，本局不能继续排位。',
  run_disqualified: '本局验证未通过，已保留记录，不能继续排位。',
  slot_exists: '此浏览器的该变体已有进行中对局，但没有找到本地存档。可以明确重开，无法从服务器恢复。',
  writer_changed: '对局写入权已变化，请重新检查本地对局。',
  local_writer_conflict: '另一页面已修改本地进度，请重新进入。',
  browser_lock_unavailable: '浏览器不支持安全的多标签页锁，请使用较新的浏览器。',
  spawn_mismatch: '出数记录验证不一致，本局不能继续排位。',
  monitoring_required: '高分记录缺少越线后的联网校验，本局不能继续排位。',
  local_storage_failed: '本地保存失败，已暂停操作。请检查浏览器存储空间。',
};
const fatalCodes = new Set(['rollback_detected', 'prefix_conflict', 'run_disqualified', 'spawn_mismatch', 'invalid_move', 'monitoring_required']);
const now = (() => { const wall = Date.now(); const start = performance.now(); return () => Math.round(wall + performance.now() - start); })();

export function useHumanSession(user, policies) {
  const run = shallowRef(null); const variant = ref('4x4'); const gate = ref('loading');
  const transition = shallowRef(null);
  const victory = ref(0);
  function continueAfterVictory() { victory.value = 0; }
  const busy = ref(false); const moveBusy = ref(false);
  const error = ref(''); const archiveNotice = ref(''); const savedSeq = ref(0);
  const archiveFailures = shallowRef([]), reportedArchiveFailures = new Set();
  let browser = ''; const writer = crypto.randomUUID(); let release; let events = new EventBuffer();
  let permitEnd = 0; let lastUpload = 0; let lastContact = 0; let timer; let disposed = false; let missingId = null;
  let pendingVisibilityCheck = false;
  let stateQueue = Promise.resolve(), networkJob = null, generation = 0;
  let currentMove = Promise.resolve();
  const waitForMove = () => currentMove;
  let connectionGraceEnd = 0, uploadBackoff = 0;
  const sameSession = context => !disposed && context.generation === generation && run.value?.id === context.id;
  const contextNow = () => ({ id: run.value?.id, generation });
  const account = () => user.value?.id || 'guest';
  const slot = () => `${account()}:${browser}:${variant.value}`;
  const policy = () => policies.value?.variants.find(v => v.id === variant.value);
  const high = (value = run.value) => value && !value.guest && (value.monitored || value.score > value.threshold);
  const explain = e => messages[e.code || e.message] || (e.status === 401 ? '登录已失效，请重新登录后检查本地进度。' : '无法连接服务器，请联网后重试。本地棋盘保持不变。');
  const recordError = e => {
    error.value = explain(e);
    if (e.code === 'slot_exists') {
      missingId = e.detail?.active_id || null;
      gate.value = 'missing';
    } else gate.value = fatalCodes.has(e.code || e.message) ? 'rejected' : 'network';
    permitEnd = 0;
  };
  async function commit(value, event) {
    try {
      await storage.saveRun(value, event ? { event, expectedSeq: value.seq - 1 } : {});
      if (event) transition.value = { fromBoard: [...run.value.board], toBoard: [...value.board], direction: event[0] & 3 };
      else if (!run.value || run.value.id !== value.id) transition.value = null;
      run.value = value; savedSeq.value = value.seq;
    } catch (e) { gate.value = 'storage'; error.value = messages.local_storage_failed; throw e; }
  }
  function inStateQueue(task) {
    const pending = stateQueue.then(task);
    stateQueue = pending.catch(() => {});
    return pending;
  }
  const save = (value, event) => inStateQueue(() => commit(value, event));
  async function applyReceipt(receipt, uploaded, context, sentAt) {
    if (!sameSession(context)) return;
    // Account for request time instead of extending an old permit on a slow response.
    permitEnd = sentAt + Math.max(0, (receipt.permit_until - receipt.server_time) * 1000 - 500);
    if (receipt.monitored) connectionGraceEnd = 0;
    lastContact = performance.now(); if (uploaded) lastUpload = lastContact;
    await inStateQueue(async () => {
      if (!sameSession(context)) return;
      // Merge only acknowledgement fields into the latest local state. Never save
      // the board/score/sequence snapshot captured when the upload began.
      await commit({ ...run.value, epoch: receipt.epoch, writer,
        permit: receipt.permit || '', monitored: receipt.monitored, serverSeq: receipt.seq, eligibility: receipt.eligibility });
    });
  }
  function background(task, { grace = false } = {}) {
    if (networkJob?.generation === generation) return networkJob.promise;
    const context = contextNow();
    const job = { ...context, promise: null };
    if (grace && navigator.onLine) connectionGraceEnd = performance.now() + 8000;
    job.promise = (async () => {
      try { await task(context); }
      catch (e) {
        if (e.status === 429 && sameSession(context)) {
          uploadBackoff = performance.now() + (e.retryAfter || 2) * 1000;
          if (performance.now() < permitEnd) return;
        }
        if (sameSession(context) && gate.value !== 'storage' && (!run.value.reason || fatalCodes.has(e.code))) recordError(e);
      }
      finally { if (networkJob === job) networkJob = null; }
    })();
    networkJob = job;
    return job.promise;
  }
  function firstOverSequence(snapshot, frozenEvents) {
    if (snapshot.firstOverSeq) return snapshot.firstOverSeq;
    let state = { ...engine.initialState(snapshot.id, snapshot.variant, snapshot.seed), variant: snapshot.variant };
    for (let index = 0; index < frozenEvents.length; index++) {
      const event = typeof frozenEvents.replayAt === 'function' ? frozenEvents.replayAt(index) : frozenEvents[index];
      state = engine.nextMove(state, event[0] & 3, event[1]).state;
      if (state.score > snapshot.threshold) return state.seq;
    }
    return snapshot.seq;
  }
  async function synchronize(context, action, snapshot, frozenEvents, status, keepalive = false) {
    const sentAt = performance.now();
    const result = await upload(snapshot, frozenEvents, browser, writer, action, status, keepalive);
    await applyReceipt(result, true, context, sentAt);
    return result;
  }
  async function checkConnection(context, snapshot, frozenEvents) {
    let status = await getStatus(snapshot, browser);
    if (!sameSession(context)) return;
    connectionGraceEnd = performance.now() + 8000;
    if (status.status !== 'active') throw Object.assign(new Error('run_disqualified'), { code: 'run_disqualified' });
    if (high(snapshot) || status.monitored) {
      if (!status.monitored && snapshot.score > snapshot.threshold) {
        // An interrupted first upload is retried at the exact crossing, followed by
        // the local tail. The server still rejects a skipped first checkpoint.
        const crossing = firstOverSequence(snapshot, frozenEvents);
        status = await json(`/api/human/runs/${snapshot.id}/writer`, { method: 'POST',
          body: { browser, writer, epoch: status.epoch } });
        if (!sameSession(context)) return;
        connectionGraceEnd = performance.now() + 8000;
        status = await synchronize(context, 'monitor', { ...snapshot, seq: crossing }, frozenEvents.slice(0, crossing), status);
        if (!sameSession(context)) return;
      }
      // Freeze the pre-check sequence: moves made while waiting cannot conceal a rollback.
      await synchronize(context, 'reentry', snapshot, frozenEvents, status);
    } else {
      const sentAt = performance.now();
      const result = await json(`/api/human/runs/${snapshot.id}/writer`, { method: 'POST',
        body: { browser, writer, epoch: status.epoch } });
      await applyReceipt(result, false, context, sentAt);
    }
    if (sameSession(context) && gate.value === 'ready') error.value = '';
  }
  async function retry() {
    if (busy.value || !run.value || ['rejected', 'storage'].includes(gate.value)) return;
    if (!navigator.onLine && high()) { offline(); return; }
    pendingVisibilityCheck = false;
    if (run.value.guest || run.value.reason) return;
    const snapshot = { ...run.value }, frozenEvents = events.clone();
    gate.value = 'ready';
    connectionGraceEnd = performance.now() + 8000;
    if (networkJob?.generation === generation) {
      const context = contextNow();
      await networkJob.promise;
      if (!sameSession(context) || ['rejected', 'storage'].includes(gate.value)) return;
    }
    if (run.value.reason) return;
    return background(context => checkConnection(context, snapshot, frozenEvents), { grace: true });
  }
  function notifyArchiveFailure(item, localEvents = null) {
    if (disposed || !item || item.guest || item.reason !== 'game_over' || item.archived
      || item.userId !== user.value?.id || reportedArchiveFailures.has(item.id)) return;
    reportedArchiveFailures.add(item.id);
    // Retain an in-memory fallback only for the current game. Older failures can
    // read IndexedDB on demand instead of keeping all historical replays in RAM.
    const retained = run.value?.id === item.id ? (localEvents || events) : null;
    archiveFailures.value = [...archiveFailures.value, { run: { ...item }, events: retained?.map(e => [...e]) || null }];
  }
  function dismissArchiveFailure(id) {
    archiveFailures.value = archiveFailures.value.filter(item => item.run.id !== id);
  }
  async function failedReplayEvents(failure) {
    if (failure.run.userId !== user.value?.id) throw new Error('wrong_account');
    return failure.events || storage.readEvents(failure.run.id);
  }
  async function flushArchives() {
    const owner = user.value?.id;
    if (!owner) return;
    if (!navigator.onLine) notifyArchiveFailure(run.value);
    try {
      const pending = (await storage.pendingArchives(owner)).filter(needsReplayUpload);
      if (user.value?.id !== owner) return;
      if (!navigator.onLine) {
        pending.forEach(item => notifyArchiveFailure(item));
        archiveNotice.value = pending.some(item => item.reason === 'game_over')
          ? '历史回放仍在本地，待补传：连接暂不可用' : '';
        return;
      }
      // A just-finished game must not wait behind older failing archive uploads.
      const currentId = run.value?.id;
      pending.sort((a, b) => Number(b.id === currentId) - Number(a.id === currentId));
      let failure = null;
      for (let item of pending) {
        if (user.value?.id !== owner) return;
        let unlock, localEvents;
        try {
          unlock = await storage.acquireSlot(`archive:${item.id}`);
          if (!unlock) continue;
          // The final tail must follow an in-flight first checkpoint/append.
          if (networkJob?.id === item.id) await networkJob.promise;
          item = await storage.readRun(item.id);
          if (!needsReplayUpload(item)) continue;
          localEvents = await storage.readEvents(item.id);
          let status = await getStatus(item, browser);
          if (status.status !== 'sealed' && !status.monitored && item.score > item.threshold) {
            // A failed first checkpoint may leave later locally saved moves. Even
            // after restart, archive the exact crossing before sending that tail.
            const crossing = firstOverSequence(item, localEvents);
            status = await upload({ ...item, seq: crossing }, localEvents.slice(0, crossing),
              browser, item.writer, 'monitor', status);
          }
          const sealed = status.status === 'sealed' ? status : await upload(item, localEvents, browser, item.writer, 'seal', status);
          const confirmed = { archived: true, serverSeq: sealed.seq, epoch: sealed.epoch, eligibility: sealed.eligibility };
          await inStateQueue(async () => {
            const latest = run.value?.id === item.id ? run.value : await storage.readRun(item.id);
            await storage.saveRun({ ...latest, ...confirmed });
            if (run.value?.id === item.id) run.value = { ...run.value, ...confirmed };
          });
          dismissArchiveFailure(item.id);
        } catch (e) {
          // Non-natural endings are best-effort archives: no banner or popup.
          if (item?.reason === 'game_over') { failure = e; notifyArchiveFailure(item, localEvents); }
        }
        finally { unlock?.(); }
      }
      if (user.value?.id === owner) archiveNotice.value = failure
        ? `历史回放仍在本地，待补传：${messages[failure.code] || '连接暂不可用'}` : '';
    } catch (e) {
      if (user.value?.id !== owner) return;
      if (run.value?.reason === 'game_over' && needsReplayUpload(run.value)) {
        notifyArchiveFailure(run.value);
        archiveNotice.value = `历史回放仍在本地，待补传：${messages[e.code] || '连接暂不可用'}`;
      }
    }
  }
  async function createLocked(replaceId = null) {
    const key = `pending:${slot()}`;
    let pending = await storage.meta(key);
    if (!pending) { pending = { request_id: crypto.randomUUID(), writer }; await storage.meta(key, pending); }
    let descriptor;
    if (user.value) {
      descriptor = await json('/api/human/runs', { method: 'POST', body: { ...pending, browser,
        variant: variant.value, replace_id: replaceId } });
    } else {
      const seed = Array.from(crypto.getRandomValues(new Uint32Array(4)), v => (v || 1).toString(16).padStart(8, '0')).join('');
      descriptor = { run_id: crypto.randomUUID(), seed, threshold: Number.MAX_SAFE_INTEGER, epoch: 1 };
    }
    const hash = await engine.initialHash(descriptor.run_id, variant.value, descriptor.seed);
    const initial = engine.initialState(descriptor.run_id, variant.value, descriptor.seed);
    const value = { ...initial,
      id: descriptor.run_id, variant: variant.value, userId: account(), browser, seed: descriptor.seed,
      initialHash: hash, hash, threshold: descriptor.threshold, epoch: descriptor.epoch, writer: pending.writer,
      guest: !user.value, monitored: false, serverSeq: 0, firstMoveAt: null, lastActionAt: null, nodesVersion: 1,
      timerSplits: timerSplitsFor(variant.value), splitTimes: {},
      fourCount: initial.board.filter(value => value === 4).length, spawnCount: 2 };
    events = new EventBuffer(); victory.value = 0; await save(value); await storage.meta(`slot:${slot()}`, value.id); await storage.meta(key, null);
    missingId = null; gate.value = 'ready'; error.value = '';
    if (user.value && pending.writer !== writer) {
      const snapshot = { ...run.value };
      await background(context => checkConnection(context, snapshot, []));
    }
  }
  async function activate(id = variant.value) {
    if (busy.value || disposed) return;
    busy.value = true; gate.value = 'loading'; error.value = ''; victory.value = 0;
    generation += 1;
    await stateQueue;
    connectionGraceEnd = 0;
    release?.(); release = null; permitEnd = 0; missingId = null; variant.value = id; run.value = null; events = new EventBuffer();
    try {
      browser ||= await storage.browserId();
      release = await storage.acquireSlot(slot());
      if (!release) { gate.value = 'other-tab'; return; }
      const localId = await storage.meta(`slot:${slot()}`); const local = await storage.readRun(localId);
      if (local) {
        run.value = local; events = new EventBuffer(await storage.readEvents(local.id)); savedSeq.value = local.seq;
        if (events.length !== local.seq) throw new Error('local_storage_failed');
        if (!Number.isInteger(local.fourCount) || !Number.isInteger(local.spawnCount)) {
          const initial = engine.initialState(local.id, local.variant, local.seed);
          const fourCount = initial.board.filter(value => value === 4).length + events.countCodeMask(64);
          await save({ ...local, fourCount, spawnCount: 2 + events.length });
        }
        if (local.nodesVersion !== 1) {
          // Recover newly displayed early milestones from this browser's own replay only.
          // Never replace board, score, sequence or RNG with server-side state.
          const replay = engine.buildReplay({ header: { run_id: local.id, variant: local.variant, seed: local.seed }, events: events.slice() });
          await save({ ...local, nodes: replay.final.nodes, nodesVersion: 1 });
        }
        if (!Array.isArray(local.timerSplits) || !local.splitTimes) {
          const timerSplits = timerSplitsFor(local.variant);
          await save({ ...run.value, timerSplits, splitTimes: engine.rebuildTimerSplitTimes(local.id, local.variant, local.seed, events, timerSplits) });
        }
        if (local.reason) { gate.value = 'ended'; void flushArchives(); return; }
        if (local.guest) { gate.value = 'ready'; return; }
        gate.value = 'ready';
        const snapshot = { ...run.value }, frozenEvents = events.clone();
        if (!navigator.onLine && high()) offline();
        else void background(async context => {
          try { await checkConnection(context, snapshot, frozenEvents); }
          catch (e) {
            if (!high(snapshot) && !high() && !e.status && sameSession(context)) {
              error.value = '当前低分局可离线游玩，超过阈值后必须联网。';
            } else throw e;
          }
        }, { grace: true });
      } else await createLocked();
    } catch (e) {
      if (e.message === 'local_storage_failed') { gate.value = 'storage'; error.value = messages.local_storage_failed; }
      else recordError(e);
    } finally { busy.value = false; }
  }
  async function play(direction) {
    if (busy.value || victory.value || gate.value !== 'ready' || !run.value || run.value.reason || document.hidden) return;
    if (high() && (performance.now() >= Math.max(permitEnd, connectionGraceEnd) || !navigator.onLine)) { offline(); return; }
    let finishMove;
    currentMove = new Promise(resolve => { finishMove = resolve; });
    moveBusy.value = true;
    busy.value = true;
    const context = contextNow();
    try {
      await inStateQueue(async () => {
        if (!sameSession(context) || gate.value !== 'ready') return;
        // RPL1 reserves 0xffffffff for an unknown timing, so the largest exact
        // interval is one millisecond smaller.
        const stamp = now(); const delta = run.value.seq ? Math.max(0, Math.min(0xfffffffe, stamp - run.value.lastActionAt)) : 0;
        const next = engine.nextMove(run.value, direction, delta);
        if (!next) return;
        next.event.push(await engine.eventHash(run.value.hash, next.event));
        next.state.hash = next.event[2]; next.state.lastActionAt = stamp;
        next.state.fourCount = (run.value.fourCount || 0) + ((next.event[0] & 64) ? 1 : 0);
        next.state.spawnCount = (run.value.spawnCount || 2) + 1;
        next.state.firstMoveAt ||= stamp;
        const won = reachedVictory(run.value, next.state);
        if (won) next.state.victoryShown = true;
        if (!high() && high(next.state)) {
          next.state.firstOverSeq = next.state.seq;
          connectionGraceEnd = navigator.onLine ? performance.now() + 8000 : 0;
        }
        // The live publisher watches run.seq. Append first so the matching
        // event is already present when that reactive update is delivered.
        events.push(next.event);
        try { await commit(next.state, next.event); }
        catch (error) { events.pop(); throw error; }
        if (engine.isOver(run.value.board, variant.value)) {
          await commit({ ...run.value, reason: 'game_over' }); gate.value = 'ended';
        } else if (won) victory.value = won;
      });
    } catch (e) {
      if (gate.value !== 'storage') recordError(e);
      if (run.value && engine.isOver(run.value.board, run.value.variant)) notifyArchiveFailure({ ...run.value, reason: 'game_over' });
    }
    finally { busy.value = false; moveBusy.value = false; finishMove(); }
    if (!sameSession(context)) return;
    if (run.value.reason) void flushArchives();
    else if (high() && !navigator.onLine) offline();
    else if (high() && !run.value.monitored) startMonitoring();
    else if (high() && run.value.seq - run.value.serverSeq >= 32 && performance.now() >= uploadBackoff) sync();
  }
  function startMonitoring() {
    if (networkJob?.generation === generation) return;
    const snapshot = { ...run.value }, crossing = firstOverSequence(snapshot, events);
    const frozenEvents = events.clone(crossing);
    return background(context => synchronize(context, 'monitor', { ...snapshot, seq: crossing }, frozenEvents,
      { seq: snapshot.serverSeq || 0, epoch: snapshot.epoch }));
  }
  function sync(keepalive = false) {
    if (!run.value || !high() || run.value.reason) return;
    if (networkJob?.generation === generation) return networkJob.promise;
    if (!run.value.monitored) return startMonitoring();
    const snapshot = { ...run.value }, frozenEvents = events.clone();
    const status = { seq: snapshot.serverSeq || 0, epoch: snapshot.epoch };
    return background(context => synchronize(context, 'append', snapshot, frozenEvents, status, keepalive));
  }
  async function tick() {
    if (disposed || busy.value || document.hidden) return;
    if (pendingVisibilityCheck) {
      pendingVisibilityCheck = false;
      if (high() && !run.value?.reason) await retry();
      return;
    }
    if (!high() || run.value?.reason || gate.value !== 'ready') return;
    if (!navigator.onLine || performance.now() >= Math.max(permitEnd, connectionGraceEnd)) { offline(); return; }
    if (networkJob?.generation === generation) return;
    if (!run.value.monitored) { startMonitoring(); return; }
    const due = performance.now() - lastUpload >= 20000 && run.value.seq > run.value.serverSeq;
    if ((due || run.value.seq - run.value.serverSeq >= 32) && performance.now() >= uploadBackoff) sync();
    else if (performance.now() - lastContact >= 5000) void background(async context => {
      const snapshot = { ...run.value }, sentAt = performance.now();
      const result = await json(`/api/human/runs/${snapshot.id}/online-check`, { method: 'POST',
        body: { browser, writer, epoch: snapshot.epoch, permit: snapshot.permit || '' } });
      await applyReceipt(result, false, context, sentAt);
    });
  }
  async function restart() {
    if (busy.value || !release) return;
    busy.value = true;
    try {
      const previous = missingId || run.value?.id;
      // Finish this run's writer/checkpoint request before releasing its server slot.
      // Historical full replay uploads do not delay starting the replacement run.
      if (networkJob?.generation === generation) await networkJob.promise;
      if (run.value && !run.value.reason) await inStateQueue(() => commit({ ...run.value, reason: 'restarted' }));
      generation += 1; connectionGraceEnd = 0; permitEnd = 0;
      void flushArchives();
      // Clear a failed creation request only for an explicit user restart.
      await storage.meta(`pending:${slot()}`, null);
      await createLocked(previous);
    } catch (e) { recordError(e); }
    finally { busy.value = false; }
  }
  function pause() { if (gate.value === 'ready') gate.value = 'paused'; }
  async function resume() {
    if (high()) await retry(); else if (gate.value === 'paused') gate.value = 'ready';
  }
  function offline() { if (high() && !run.value?.reason) { permitEnd = 0; connectionGraceEnd = 0; gate.value = 'network'; error.value = '高分对局需要联网，连接恢复后可继续。'; } }
  function online() { if (gate.value === 'network') retry(); flushArchives(); }
  function visibility() {
    if (document.hidden) {
      if (high() && !run.value?.reason) pendingVisibilityCheck = true;
      if (high() && !run.value?.reason && gate.value === 'ready') gate.value = 'checking';
      if (high() && !run.value?.reason) sync(true);
      permitEnd = 0; connectionGraceEnd = 0;
    } else if (high() && !run.value?.reason) retry();
  }
  function start() {
    disposed = false;
    if (timer) clearInterval(timer);
    window.removeEventListener('offline', offline); window.removeEventListener('online', online);
    document.removeEventListener('visibilitychange', visibility);
    timer = setInterval(tick, 500);
    window.addEventListener('offline', offline); window.addEventListener('online', online);
    document.addEventListener('visibilitychange', visibility);
  }
  function liveCheckpoint() {
    if (!run.value || run.value.reason || run.value.guest || high()) return;
    if (networkJob?.generation === generation) return networkJob.promise;
    const snapshot = { ...run.value };
    if (![32768, 65536].some(value => snapshot.nodes?.[value]?.seq === snapshot.seq)) return;
    const frozenEvents = events.clone();
    const status = { seq: snapshot.serverSeq || 0, epoch: snapshot.epoch };
    return background(context => synchronize(context, 'live', snapshot, frozenEvents, status));
  }
  function stop() {
    if (disposed) return;
    disposed = true; generation += 1; clearInterval(timer); release?.();
    timer = null; release = null;
    window.removeEventListener('offline', offline); window.removeEventListener('online', online);
    document.removeEventListener('visibilitychange', visibility);
  }
  return { run, variant, gate, victory, continueAfterVictory, busy, moveBusy, waitForMove, error, archiveNotice, archiveFailures, dismissArchiveFailure, failedReplayEvents, savedSeq, transition, activate, play, retry, restart,
    pause, resume, start, stop, flushArchives, high, now, liveCheckpoint,
    getEventCount: () => events.length,
    getEvents: (start = 0, end = events.length) => events.slice(start, end).map(e => [...e]), getPolicy: policy,
    liveContext: () => ({ browser, writer, run: run.value ? { ...run.value } : null }) };
}
