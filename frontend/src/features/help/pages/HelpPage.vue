<template>
  <div class="page-root">
    <div class="flex h-full min-h-0 w-full max-w-7xl flex-col overflow-hidden border border-border-main bg-bg-main shadow-2xl transition-colors duration-300 md:flex-row">
      <aside class="flex max-h-[42%] w-full shrink-0 flex-col border-b border-border-main bg-bg-card md:h-full md:max-h-none md:w-72 md:border-b-0 md:border-r">
        <div class="border-b border-border-main bg-btn-bg/5 p-4 sm:p-5">
          <div class="flex items-center justify-between gap-3">
            <h2 class="ui-metric flex min-w-0 items-center gap-2 font-black uppercase text-text-main">
              <span class="h-6 w-2 rounded-full accent-icon-prominent"></span>
              <span class="truncate">{{ $t('tabs.help') }}</span>
            </h2>
            <button
              type="button"
              class="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border-main bg-bg-main/85 text-text-secondary transition-colors hover:border-accent/45 hover:text-text-main"
              @click="openSearch"
              :title="$t('help.search.open')"
              :aria-label="$t('help.search.open')"
            >
              <svg viewBox="0 0 20 20" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="8.5" cy="8.5" r="4.75" />
                <path d="M12.2 12.2L16.2 16.2" />
              </svg>
            </button>
          </div>

          <UiSelect
            v-model="selectedDocumentId"
            class="mt-4"
            :options="documentOptions"
            :aria-label="$t('help.documents.select')"
            trigger-class="min-h-11 rounded-lg border border-border-main bg-bg-main/90 px-3 py-2"
            menu-class="min-w-full"
            option-class="rounded-lg"
          />

          <div class="mt-3 flex items-center gap-2">
            <button type="button" class="action-btn-small" @click="setAllTocNodes(true)">
              {{ $t('help.toc.expandAll') }}
            </button>
            <button type="button" class="action-btn-small" @click="setAllTocNodes(false)">
              {{ $t('help.toc.collapseAll') }}
            </button>
          </div>
        </div>

        <nav class="custom-scrollbar flex-1 space-y-1 overflow-y-auto p-2" :aria-label="$t('help.toc.label')">
          <template v-if="visibleTocItems.length > 0">
            <div
              v-for="item in visibleTocItems"
              :key="item.id"
              class="group flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left transition-colors hover:bg-accent/10"
              :style="{ paddingLeft: `${0.75 + item.depth * 0.85}rem` }"
            >
              <button
                v-if="item.hasChildren"
                type="button"
                class="flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-text-secondary transition-colors hover:bg-bg-main hover:text-text-main"
                @click.stop="toggleTocNode(item.id)"
                :aria-label="item.isOpen ? $t('help.toc.collapseAll') : $t('help.toc.expandAll')"
              >
                <svg
                  viewBox="0 0 20 20"
                  class="h-3.5 w-3.5 transition-transform duration-200"
                  :class="item.isOpen ? 'rotate-90' : ''"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="1.9"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                >
                  <path d="M7 4.5L13 10L7 15.5" />
                </svg>
              </button>
              <span v-else class="h-6 w-6 shrink-0"></span>
              <button
                type="button"
                class="min-w-0 flex-1 text-left transition-colors group-hover:text-accent"
                :class="{
                  'ui-text-lg font-black text-text-main': item.level === 1,
                  'ui-body font-bold text-text-secondary': item.level === 2,
                  'ui-control font-bold text-text-secondary/70': item.level === 3,
                  'ui-control italic text-text-secondary/45': item.level >= 4
                }"
                @click="scrollTo(item.id)"
              >
                {{ item.text }}
              </button>
            </div>
          </template>
          <div v-else class="p-6 text-center font-bold italic text-text-secondary/40">
            {{ loading ? '...' : $t('help.toc.empty') }}
          </div>
        </nav>
      </aside>

      <section class="relative flex min-h-0 flex-1 flex-col overflow-hidden bg-bg-main">
        <div
          v-if="searchOpen"
          class="absolute right-4 top-4 z-20 w-[min(520px,calc(100%-2rem))] rounded-lg border border-border-main bg-bg-card/96 p-3 shadow-[0_18px_45px_rgba(15,23,42,0.18)] backdrop-blur-md sm:right-8 sm:w-[min(520px,calc(100%-4rem))]"
        >
          <div class="flex flex-wrap items-center gap-2 sm:flex-nowrap">
            <input
              ref="searchInput"
              v-model="searchQuery"
              type="text"
              class="min-w-[12rem] flex-1 rounded-lg border border-border-main bg-bg-main px-4 py-2.5 ui-body font-bold text-text-main outline-none transition-colors focus:border-accent"
              :placeholder="$t('help.search.placeholder')"
            />
            <div class="rounded-lg border border-border-main bg-bg-main/85 px-3 py-2 ui-control font-black tabular-nums text-text-secondary">
              {{ searchMatchCount > 0 ? `${activeSearchIndex + 1}/${searchMatchCount}` : '0/0' }}
            </div>
            <button type="button" class="action-btn-small" @click="previousMatch" :title="$t('help.search.previous')" aria-label="Previous">↑</button>
            <button type="button" class="action-btn-small" @click="nextMatch" :title="$t('help.search.next')" aria-label="Next">↓</button>
            <button type="button" class="action-btn-small" @click="closeSearch">{{ $t('common.close') }}</button>
          </div>
        </div>

        <div v-if="loading" class="absolute inset-0 z-10 flex items-center justify-center bg-bg-main/80 backdrop-blur-sm">
          <div class="flex flex-col items-center gap-4">
            <div class="analysis-loader-ring h-12 w-12 animate-spin rounded-full border-4"></div>
            <p class="ui-body animate-pulse font-black text-text-main">{{ $t('help.documents.loading') }}</p>
          </div>
        </div>

        <div ref="contentArea" class="custom-scrollbar flex-1 overflow-y-auto scroll-smooth bg-bg-main p-4 sm:p-8 lg:p-12">
          <article ref="articleRef" class="help-article max-w-none">
            <div v-if="isManualSelected" class="markdown-body prose prose-slate max-w-none" v-html="manualHtmlContent"></div>
            <GuideDocumentView
              v-else-if="activeGuideDocument"
              :document="activeGuideDocument"
              :open-board-label="$t('help.guide.openBoard')"
              @open-board="handleOpenBoard"
            />
            <div v-else-if="guideLoadError" class="guide-error" role="alert">
              <h3>{{ $t('help.documents.loadError') }}</h3>
              <p>{{ guideLoadError }}</p>
            </div>
          </article>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, ref, toRef, watch } from 'vue';
import { useI18n } from 'vue-i18n';

import { TAB_IDS } from '../../../app/tabRegistry';
import UiSelect from '../../../components/UiSelect.vue';
import GuideDocumentView from '../components/GuideDocumentView.vue';
import { useHelpSession } from '../composables/useHelpSession';
import { loadGuideDocument, loadGuideIndex } from '../services/guideLibrary';
import { createGuideTrainerJumpDetail } from '../utils/guideNavigation';

const MANUAL_DOCUMENT_ID = 'manual';

const props = defineProps({
  active: {
    type: Boolean,
    default: false,
  },
});

const emit = defineEmits(['navigate-tab']);
const { t } = useI18n();

const {
  htmlContent: manualHtmlContent,
  toc: manualToc,
  loading: manualLoading,
  contentArea,
  articleRef,
  searchInput,
  searchOpen,
  searchQuery,
  searchMatchCount,
  activeSearchIndex,
  scrollTo,
  openSearch,
  closeSearch,
  nextMatch,
  previousMatch,
  refreshSearch,
  typesetMath,
} = useHelpSession(toRef(props, 'active'));

const selectedDocumentId = ref(MANUAL_DOCUMENT_ID);
const guideEntries = ref([]);
const activeGuideDocument = ref(null);
const guideRegistryLoading = ref(false);
const guideLoading = ref(false);
const guideLoadError = ref('');
const tocOpenState = ref({});
const guideCache = new Map();

const isManualSelected = computed(() => selectedDocumentId.value === MANUAL_DOCUMENT_ID);
const loading = computed(() => (
  isManualSelected.value ? manualLoading.value : guideLoading.value
));
const activeToc = computed(() => (
  isManualSelected.value ? manualToc.value : activeGuideDocument.value?.toc || []
));
const documentOptions = computed(() => [
  { value: MANUAL_DOCUMENT_ID, label: t('help.documents.manual') },
  ...guideEntries.value.map((entry) => ({
    value: entry.id,
    label: entry.title,
    badge: entry.language?.toUpperCase() || '',
  })),
]);

const buildTocTree = (items) => {
  const root = [];
  const stack = [];

  for (const item of items || []) {
    const node = { ...item, children: [] };
    while (stack.length && stack[stack.length - 1].level >= node.level) {
      stack.pop();
    }
    if (stack.length) {
      stack[stack.length - 1].children.push(node);
    } else {
      root.push(node);
    }
    stack.push(node);
  }

  return root;
};

const tocTree = computed(() => buildTocTree(activeToc.value));

const ensureOpenState = (nodes) => {
  for (const node of nodes) {
    if (node.children.length > 0 && tocOpenState.value[node.id] === undefined) {
      tocOpenState.value[node.id] = true;
    }
    ensureOpenState(node.children);
  }
};

watch(tocTree, ensureOpenState, { immediate: true });

const flattenVisibleNodes = (nodes, depth = 0) => {
  const result = [];
  for (const node of nodes) {
    const hasChildren = node.children.length > 0;
    const isOpen = hasChildren ? tocOpenState.value[node.id] !== false : false;
    result.push({ ...node, depth, hasChildren, isOpen });
    if (hasChildren && isOpen) {
      result.push(...flattenVisibleNodes(node.children, depth + 1));
    }
  }
  return result;
};

const visibleTocItems = computed(() => flattenVisibleNodes(tocTree.value));

const toggleTocNode = (id) => {
  tocOpenState.value[id] = !(tocOpenState.value[id] !== false);
};

const setAllTocNodes = (expanded) => {
  const visit = (nodes) => {
    for (const node of nodes) {
      if (node.children.length > 0) {
        tocOpenState.value[node.id] = expanded;
      }
      visit(node.children);
    }
  };
  visit(tocTree.value);
};

const ensureGuideRegistry = async () => {
  if (guideEntries.value.length || guideRegistryLoading.value) {
    return;
  }
  guideRegistryLoading.value = true;
  try {
    guideEntries.value = await loadGuideIndex();
  } catch (error) {
    console.error('Failed to load guide registry.', error);
  } finally {
    guideRegistryLoading.value = false;
  }
};

const showSelectedDocument = async (documentId) => {
  closeSearch();
  tocOpenState.value = {};
  contentArea.value?.scrollTo({ top: 0 });
  guideLoadError.value = '';

  if (documentId === MANUAL_DOCUMENT_ID) {
    activeGuideDocument.value = null;
    await nextTick();
    typesetMath();
    refreshSearch();
    return;
  }

  await ensureGuideRegistry();
  const entry = guideEntries.value.find((candidate) => candidate.id === documentId);
  if (!entry) {
    guideLoadError.value = t('help.documents.notFound');
    return;
  }

  if (guideCache.has(documentId)) {
    activeGuideDocument.value = guideCache.get(documentId);
    await nextTick();
    refreshSearch();
    return;
  }

  guideLoading.value = true;
  try {
    const document = await loadGuideDocument(entry);
    if (selectedDocumentId.value !== documentId) {
      return;
    }
    guideCache.set(documentId, document);
    activeGuideDocument.value = document;
    await nextTick();
    refreshSearch();
  } catch (error) {
    console.error('Failed to load guide document.', error);
    guideLoadError.value = error instanceof Error ? error.message : String(error);
  } finally {
    guideLoading.value = false;
  }
};

const handleOpenBoard = (board, documentId, trainerContext) => {
  const detail = createGuideTrainerJumpDetail(board, documentId, trainerContext);
  if (detail) {
    emit('navigate-tab', TAB_IDS.TRAINER, detail);
  }
};

watch(
  () => props.active,
  (active) => {
    if (active) {
      ensureGuideRegistry();
    }
  },
  { immediate: true },
);

watch(selectedDocumentId, showSelectedDocument);
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 6px;
}

.custom-scrollbar::-webkit-scrollbar-track {
  background: var(--scrollbar-track);
}

.custom-scrollbar::-webkit-scrollbar-thumb {
  border-radius: 10px;
  background: var(--scrollbar-thumb);
}

.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background: var(--scrollbar-thumb-hover);
}

.analysis-loader-ring {
  border-color: color-mix(in srgb, var(--accent) 20%, transparent);
  border-top-color: var(--accent);
  box-shadow: 0 0 18px color-mix(in srgb, var(--accent) 18%, transparent);
}

.guide-error {
  border-left: 4px solid #dc2626;
  background: color-mix(in srgb, #dc2626 8%, var(--bg-card));
  padding: 1rem 1.25rem;
  color: var(--text-main);
}

.guide-error h3 {
  margin-bottom: 0.35rem;
  font-weight: 900;
}

:deep(.markdown-body) {
  color: var(--text-main);
  font-family: var(--font-stack-system);
  line-height: 1.7;
}

:deep(.markdown-body h1) {
  margin: 2.5rem 0 1.5rem;
  border-bottom: 2px solid var(--accent);
  padding-bottom: 0.5rem;
  color: var(--text-main);
  font-size: calc(2.25rem * var(--ui-scale));
  font-weight: 900;
  letter-spacing: 0;
}

:deep(.markdown-body h2) {
  margin: 1.8rem 0 1.2rem;
  border-bottom: 1px solid var(--border-main);
  padding-bottom: 0.3rem;
  color: var(--text-main);
  font-size: calc(1.65rem * var(--ui-scale));
  font-weight: 800;
  letter-spacing: 0;
}

:deep(.markdown-body h3) {
  margin-top: 1.5rem;
  color: var(--accent);
  font-size: calc(1.2rem * var(--ui-scale));
  font-weight: 700;
  letter-spacing: 0;
}

:deep(.markdown-body p) {
  margin: 0.75rem 0 1.25rem;
  opacity: 0.9;
}

:deep(.markdown-body code) {
  border: 1px solid var(--border-main);
  border-radius: 6px;
  background: var(--bg-card);
  padding: 0.2rem 0.4rem;
  color: var(--accent);
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.9em;
}

:deep(.markdown-body pre) {
  overflow-x: auto;
  margin-bottom: 2rem;
  border: 1px solid var(--border-main);
  border-radius: 8px;
  background: var(--ctrl-bg);
  padding: 1.25rem;
  color: var(--text-main);
}

:deep(.markdown-body pre code) {
  border: 0;
  background: transparent;
  padding: 0;
  color: inherit;
}

:deep(.markdown-body table) {
  width: 100%;
  margin-bottom: 2rem;
  border-collapse: collapse;
}

:deep(.markdown-body th),
:deep(.markdown-body td) {
  border: 1px solid var(--border-main);
  padding: 0.75rem;
  text-align: left;
}

:deep(.markdown-body th) {
  background: var(--bg-card);
  font-weight: 800;
}

:deep(.markdown-body ul),
:deep(.markdown-body ol) {
  margin-bottom: 1.5rem;
  padding-left: 2rem;
}

:deep(.markdown-body ul) {
  list-style-type: disc;
}

:deep(.markdown-body ol) {
  list-style-type: decimal;
}

:deep(.markdown-body li) {
  display: list-item;
  margin-bottom: 0.5rem;
}

:deep(.markdown-body blockquote) {
  margin: 1.5rem 0;
  border-left: 4px solid var(--accent);
  background: var(--bg-card);
  padding: 1rem 1.5rem;
  font-style: italic;
}

:deep(.markdown-body a) {
  color: var(--accent);
  font-weight: 700;
  text-decoration: underline;
}

:deep(.help-search-hit) {
  border-radius: 0.25rem;
  background: color-mix(in srgb, var(--accent) 22%, transparent);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 24%, transparent);
  color: inherit;
  padding: 0.03em 0.12em;
}

:deep(.help-search-hit.is-active) {
  background: color-mix(in srgb, var(--accent) 42%, white 8%);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 16%, transparent);
}
</style>
