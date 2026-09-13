import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { computed, createSSRApp, h, ref } from 'vue';
import { renderToString } from 'vue/server-renderer';
import { createServer } from 'vite';

let vite, useGoodnessMatch, useFreeGoodnessMatch;
const originalWindow = globalThis.window;
before(async () => {
  globalThis.window = { setTimeout, clearTimeout };
  vite = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), configFile: false,
    server: { middlewareMode: true, hmr: false }, optimizeDeps: { noDiscovery: true } });
  ({ useGoodnessMatch } = await vite.ssrLoadModule('/src/features/battle/modes/goodness/useGoodnessMatch.js'));
  ({ useFreeGoodnessMatch } = await vite.ssrLoadModule('/src/features/battle/modes/freeGoodness/useFreeGoodnessMatch.js'));
  const { battleClient } = await vite.ssrLoadModule('/src/features/battle/services/battleClient.js');
  const buffer = new ArrayBuffer(51);
  const view = new DataView(buffer);
  view.setUint32(1, 0x11, true);
  for (let i = 1; i <= 2; i++) {
    view.setUint8(i * 17, 3); // Right, spawn a 2 at top left.
    [0, 0, 2e9, 3e9].forEach((value, direction) => view.setUint32(i * 17 + 1 + direction * 4, value, true));
  }
  battleClient.route = async () => ({ buffer, certaintyStep: -1 });
});
after(async () => {
  await vite?.close();
  if (originalWindow === undefined) delete globalThis.window;
  else globalThis.window = originalWindow;
});

async function harness(mode) {
  const room = ref({ room_id: 'r', room_code: 'ABCDEF', pattern: 'L3', full_pattern: 'L3_256',
    mode_key: mode, status: 'running', round: { round_id: 'round' }, route: { step_count: 2 },
    results: [{ actor_key: 'u:1', status: 'playing', route_index: 0, goodness_of_fit: 1,
      mode_data: { state_status: 'input', board_hex: '0000000000000011' } }] });
  const ownResult = computed(() => room.value.results[0]);
  let adapter, session;
  const sent = [];
  const core = { room, ownResult, error: ref(''), matchActive: ref(true), spectatorMode: ref(false),
    ownFinished: computed(() => ownResult.value.status !== 'playing'),
    isOwnActor: result => result?.actor_key === 'u:1',
    registerModeAdapter: value => { adapter = value; },
    sendModeAction: (...args) => { sent.push(args); return 'request'; } };
  await renderToString(createSSRApp({ setup() {
    session = (mode === 'goodness' ? useGoodnessMatch : useFreeGoodnessMatch)(core, ref(true), ref({ id: 1 }));
    return () => h('div');
  } }));
  await adapter.onRoomApplied(room.value);
  return { room, ownResult, adapter, session, sent };
}

for (const mode of ['goodness', 'free_goodness']) {
  test(`${mode}: confirmed forfeit cancels correction and ignores late move acknowledgements`, async () => {
    const { room, ownResult, adapter, session, sent } = await harness(mode);
    try {
      if (mode === 'goodness') assert.equal(session.submitMove('left'), true);
      else await adapter.handleMessage({ action: 'BATTLE_ACTION_ACCEPTED', data: {
        round_id: 'round', sequence: 1, corrected: true, awaiting_ack: true,
        selected_direction: 'left', executed_direction: 'right',
        previous_board_hex: '0000000000000011', board_hex: '1000000000000002',
      } });
      assert.ok(session.matchProps.value.wrongOverlay);
      Object.assign(ownResult.value, { status: 'disqualified', route_index: 1, last_sequence: 1,
        mode_data: { finish_reason: 'forfeit', state_status: 'finished', board_hex: '1000000000000002' } });
      await adapter.onRoomApplied(room.value);
      assert.equal(session.matchProps.value.wrongOverlay, null);
      const frame = session.matchProps.value.boardFrame;
      const count = sent.length;
      assert.equal(session.matchListeners['continue-correction'](), false);
      assert.equal(session.submitMove('right'), false);
      await adapter.handleMessage({ action: 'BATTLE_ACTION_ACCEPTED', data: {
        round_id: 'round', sequence: 1, complete: true, corrected: true, board_hex: '0000000000000011',
      } });
      assert.equal(ownResult.value.status, 'disqualified');
      assert.equal(session.matchProps.value.wrongOverlay, null);
      assert.equal(session.matchProps.value.boardFrame, frame);
      assert.equal(sent.length, count);
    } finally { adapter.dispose(); }
  });
}
