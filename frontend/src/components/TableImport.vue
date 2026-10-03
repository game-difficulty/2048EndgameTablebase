<template>
  <button type="button" class="action-btn-small !w-auto shrink-0" @click="openDialog">{{ $t('tables.import') }}</button>
  <Teleport to="body">
    <div v-if="open" class="fixed inset-0 z-[300] flex items-center justify-center bg-black/40 p-5 backdrop-blur-sm" @click.self="close">
      <section role="dialog" aria-modal="true" :aria-label="$t('tables.import')" class="w-full max-w-2xl rounded-2xl border border-border-main bg-bg-card p-6 text-text-main shadow-2xl" @keydown.esc.stop="close" @keydown.tab="trapFocus">
        <div class="mb-4 flex items-center justify-between gap-4">
          <h2 class="text-xl font-black">{{ $t('tables.import') }}</h2>
          <button class="action-btn-small !w-auto" :disabled="busy" @click="close">{{ $t('analysis.close') }}</button>
        </div>
        <p class="mb-4 ui-body text-text-secondary">{{ $t('tables.help') }}</p>
        <form class="flex gap-2" @submit.prevent="scan">
          <input ref="pathInput" v-model="path" :disabled="busy" :aria-label="$t('tables.folder')" class="min-w-0 flex-1 rounded-lg border border-border-main bg-bg-main px-3 py-2 outline-none focus:border-accent" :placeholder="$t('tables.folder')" />
          <button type="button" class="action-btn-small !w-auto" :disabled="busy" @click="browse">{{ $t('tables.browse') }}</button>
          <button class="action-btn-small !w-auto" :disabled="busy || !path.trim()">{{ $t('tables.scan') }}</button>
        </form>
        <div v-if="rows.length" class="mt-4 max-h-[40vh] space-y-2 overflow-y-auto">
          <label v-for="(row, index) in rows" :key="row.path + row.pattern + row.target" class="flex cursor-pointer items-start gap-3 rounded-xl border border-border-main p-3 hover:border-accent">
            <input v-model="selected" :value="index" type="checkbox" :disabled="busy" class="mt-1 accent-accent" />
            <span class="min-w-0 flex-1"><span class="font-bold">{{ row.pattern }} · {{ goalLabel(row.target) }}</span>
              <span class="ml-2 ui-caption text-text-secondary">{{ row.dtype }} · P(4) {{ row.spawn_rate }}{{ row.assumed_rate ? ' *' : '' }}</span>
              <span class="mt-1 block break-all ui-caption text-text-secondary">{{ row.path }}</span>
            </span>
          </label>
        </div>
        <p v-if="rows.some(row => row.assumed_rate)" class="mt-3 ui-caption text-text-secondary">{{ $t('tables.assumedRate') }}</p>
        <p v-if="message" role="status" class="mt-4 ui-body text-text-secondary">{{ message }}</p>
        <p v-if="error" role="alert" class="mt-4 ui-body text-red-400">{{ error }}</p>
        <div class="mt-5 flex justify-end"><button class="action-btn btn-prominent !w-auto" :disabled="busy || !selected.length" @click="register">{{ busy ? $t('common.updating') : $t('tables.add') }}</button></div>
      </section>
    </div>
  </Teleport>
</template>

<script setup>
import { nextTick, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { useAppSettingsStore } from '../app/useAppSettings';
import { tryDesktopDialog } from '../services/runtime/desktopDialogs';
import { goalLabel } from '../utils/goalTarget';
const emit = defineEmits(['imported']);
const { t } = useI18n();
const { tableRequest, config } = useAppSettingsStore();
const open = ref(false), busy = ref(false), path = ref(''), rows = ref([]), selected = ref([]), error = ref(''), message = ref(''), pathInput = ref(null);
let opener;
const openDialog = async () => { opener = document.activeElement; open.value = true; await nextTick(); pathInput.value?.focus(); };
const close = () => { if (!busy.value) { open.value = false; opener?.focus(); } };
const trapFocus = (event) => {
  const elements = [...event.currentTarget.querySelectorAll('button:not(:disabled), input:not(:disabled)')];
  const destination = event.shiftKey ? elements.at(-1) : elements[0];
  if (document.activeElement === (event.shiftKey ? elements[0] : elements.at(-1))) {
    event.preventDefault(); destination?.focus();
  }
};
const perform = async (task) => {
  busy.value = true; error.value = ''; message.value = '';
  try { await task(); } catch (exc) { error.value = exc.message; } finally { busy.value = false; }
};
const scan = () => perform(async () => {
  rows.value = []; selected.value = [];
  const result = await tableRequest('TABLE_SCAN', { path: path.value.trim() });
  rows.value = result.tables;
  selected.value = rows.value.map((_, index) => index);
  if (!rows.value.length) message.value = t('tables.empty');
});
const browse = async () => {
  await perform(async () => {
    const result = await tryDesktopDialog('select_folder');
    const folder = result.handled ? result.value : (await tableRequest('TABLE_PICK_FOLDER')).path;
    if (folder) path.value = folder;
  });
  if (path.value && !error.value) await scan();
};
const register = () => perform(async () => {
  const result = await tableRequest('TABLE_IMPORT', { tables: selected.value.map(index => rows.value[index]) });
  const matching = result.tables.filter(row => Math.abs(row.spawn_rate - Number(config.value['4_spawn_rate'])) < 1e-4);
  emit('imported', matching);
  if (matching.length) { open.value = false; opener?.focus(); }
  else message.value = t('tables.otherRate');
});
</script>
