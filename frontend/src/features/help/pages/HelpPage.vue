<template>
  <div class="page-root">
    <div class="flex h-full min-h-0 w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-border-main bg-bg-main shadow-2xl transition-colors duration-300 md:flex-row">
      <div class="flex h-1/3 w-full flex-col border-b border-border-main bg-bg-card md:h-full md:w-72 md:border-b-0 md:border-r">
        <div class="border-b border-border-main bg-btn-bg/5 p-6">
          <div class="flex items-center justify-between gap-3">
            <h2 class="ui-metric flex min-w-0 items-center gap-2 font-black uppercase tracking-tighter text-text-main">
              <span class="h-6 w-2 rounded-full accent-icon-prominent"></span>
              <span class="truncate">{{ $t('tabs.help') }}</span>
            </h2>
            <button
              type="button"
              class="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-border-main bg-bg-main/85 text-text-secondary transition-colors hover:border-accent/45 hover:text-text-main"
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
          <div class="mt-4 flex items-center gap-2">
            <button
              type="button"
              class="action-btn-small"
              @click="setAllTocNodes(true)"
            >
              {{ $t('help.toc.expandAll') }}
            </button>
            <button
              type="button"
              class="action-btn-small"
              @click="setAllTocNodes(false)"
            >
              {{ $t('help.toc.collapseAll') }}
            </button>
          </div>
        </div>

        <div class="custom-scrollbar flex-1 space-y-1 overflow-y-auto p-2">
          <template v-if="visibleTocItems.length > 0">
            <div
              v-for="item in visibleTocItems"
              :key="item.id"
              class="group flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left transition-all duration-300 hover:bg-accent/10 active:scale-95"
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
                @click="scrollTo(item.id)"
                class="min-w-0 flex-1 text-left transition-colors group-hover:text-accent"
                :class="{
                  'ui-text-lg font-black uppercase tracking-tight text-text-main': item.level === 1,
                  'ui-body font-bold text-text-secondary': item.level === 2,
                  'ui-control font-bold text-text-secondary/70': item.level === 3,
                  'ui-control italic text-text-secondary/45': item.level >= 4
                }"
              >
                {{ item.text }}
              </button>
            </div>
          </template>
          <div v-else class="p-6 text-center font-bold italic text-text-secondary/40">
            {{ loading ? '...' : 'No sections found' }}
          </div>
        </div>
      </div>

      <div class="relative flex h-2/3 flex-1 flex-col overflow-hidden bg-bg-main md:h-full">
        <div
          v-if="searchOpen"
          class="absolute left-6 right-6 top-5 z-20 rounded-2xl border border-border-main bg-bg-card/92 p-3 shadow-[0_18px_45px_rgba(15,23,42,0.18)] backdrop-blur-md md:left-auto md:right-8 md:w-[min(520px,calc(100%-4rem))]"
        >
          <div class="flex items-center gap-2">
            <div class="pointer-events-none flex h-10 w-10 min-w-[2.5rem] shrink-0 items-center justify-center rounded-xl border border-border-main/70 bg-bg-main/80 text-text-secondary">
              <svg viewBox="0 0 20 20" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="8.5" cy="8.5" r="4.75" />
                <path d="M12.2 12.2L16.2 16.2" />
              </svg>
            </div>
            <input
              ref="searchInput"
              v-model="searchQuery"
              type="text"
              class="flex-1 rounded-xl border border-border-main bg-bg-main px-4 py-2.5 ui-body font-bold text-text-main outline-none transition-colors focus:border-accent"
              :placeholder="$t('help.search.placeholder')"
            />
            <div class="rounded-xl border border-border-main bg-bg-main/85 px-3 py-2 ui-control font-black tabular-nums text-text-secondary">
              {{ searchMatchCount > 0 ? `${activeSearchIndex + 1}/${searchMatchCount}` : '0/0' }}
            </div>
            <button type="button" class="action-btn-small" @click="previousMatch" :title="$t('help.search.previous')">
              ↑
            </button>
            <button type="button" class="action-btn-small" @click="nextMatch" :title="$t('help.search.next')">
              ↓
            </button>
            <button type="button" class="action-btn-small" @click="closeSearch">
              {{ $t('common.close') }}
            </button>
          </div>
          <div class="mt-2 flex items-center justify-between gap-3 px-1">
            <div class="ui-kicker font-bold text-text-secondary">
              {{ $t('help.search.shortcutHint') }}
            </div>
            <div class="ui-kicker font-bold text-text-secondary">
              {{ $t('help.search.enterHint') }}
            </div>
          </div>
        </div>

        <div v-if="loading" class="absolute inset-0 z-10 flex items-center justify-center bg-bg-main/80 backdrop-blur-sm">
          <div class="flex flex-col items-center gap-4">
            <div class="analysis-loader-ring h-12 w-12 animate-spin rounded-full border-4"></div>
            <p class="ui-body animate-pulse font-black uppercase tracking-widest text-text-main">Loading Help...</p>
          </div>
        </div>

        <div ref="contentArea" class="custom-scrollbar flex-1 overflow-y-auto scroll-smooth bg-bg-main p-8 md:p-12">
          <article ref="articleRef" class="markdown-body prose prose-slate max-w-none" v-html="htmlContent"></article>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, toRef, watch } from 'vue';

import { useHelpSession } from '../composables/useHelpSession';

const props = defineProps({
  active: {
    type: Boolean,
    default: false,
  },
});

const {
  htmlContent,
  toc,
  loading,
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
} = useHelpSession(
  toRef(props, 'active')
);

const tocOpenState = ref({});

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

const tocTree = computed(() => buildTocTree(toc.value));

const ensureOpenState = (nodes) => {
  for (const node of nodes) {
    if (node.children.length > 0 && tocOpenState.value[node.id] === undefined) {
      tocOpenState.value[node.id] = true;
    }
    if (node.children.length > 0) {
      ensureOpenState(node.children);
    }
  }
};

watch(
  tocTree,
  (nodes) => {
    ensureOpenState(nodes);
  },
  { immediate: true }
);

const flattenVisibleNodes = (nodes, depth = 0) => {
  const result = [];

  for (const node of nodes) {
    const hasChildren = node.children.length > 0;
    const isOpen = hasChildren ? tocOpenState.value[node.id] !== false : false;
    result.push({
      ...node,
      depth,
      hasChildren,
      isOpen,
    });

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
        visit(node.children);
      }
    }
  };

  visit(tocTree.value);
};
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 6px;
}

.custom-scrollbar::-webkit-scrollbar-track {
  background: var(--scrollbar-track);
}

.custom-scrollbar::-webkit-scrollbar-thumb {
  background: var(--scrollbar-thumb);
  border-radius: 10px;
}

.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background: var(--scrollbar-thumb-hover);
}

.custom-scrollbar::-webkit-scrollbar-corner {
  background: var(--scrollbar-corner);
}

.analysis-loader-ring {
  border-color: color-mix(in srgb, var(--accent) 20%, transparent);
  border-top-color: var(--accent);
  box-shadow: 0 0 18px color-mix(in srgb, var(--accent) 18%, transparent);
}

:deep(.markdown-body) {
  color: var(--text-main);
  font-family: var(--font-stack-system);
  line-height: 1.7;
}

:deep(.markdown-body h1) {
  font-size: calc(2.5rem * var(--ui-scale));
  font-weight: 900;
  border-bottom: 2px solid var(--accent);
  padding-bottom: 0.5rem;
  margin-top: 2.5rem;
  margin-bottom: 1.5rem;
  color: var(--text-main);
  text-transform: uppercase;
  letter-spacing: -0.02em;
}

:deep(.markdown-body h2) {
  font-size: calc(1.75rem * var(--ui-scale));
  font-weight: 800;
  border-bottom: 1px solid var(--border-main);
  padding-bottom: 0.3rem;
  margin-top: 1.8rem;
  margin-bottom: 1.2rem;
  color: var(--text-main);
}

:deep(.markdown-body h3) {
  font-size: calc(1.25rem * var(--ui-scale));
  font-weight: 700;
  margin-top: 1.5rem;
  color: var(--accent);
}

:deep(.markdown-body p) {
  margin-top: 0.75rem;
  margin-bottom: 1.25rem;
  opacity: 0.9;
}

:deep(.markdown-body code) {
  background: var(--bg-card);
  color: var(--accent);
  padding: 0.2rem 0.4rem;
  border-radius: 6px;
  font-size: 0.9em;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  border: 1px solid var(--border-main);
}

:deep(.markdown-body pre) {
  background: var(--ctrl-bg);
  color: var(--text-main);
  padding: 1.25rem;
  border-radius: 12px;
  overflow-x: auto;
  margin-bottom: 2rem;
  border: 1px solid var(--border-main);
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
}

:deep(.markdown-body pre::-webkit-scrollbar) {
  width: 8px;
  height: 8px;
}

:deep(.markdown-body pre::-webkit-scrollbar-track) {
  background: var(--scrollbar-track);
}

:deep(.markdown-body pre::-webkit-scrollbar-thumb) {
  background: var(--scrollbar-thumb);
  border-radius: 10px;
}

:deep(.markdown-body pre::-webkit-scrollbar-thumb:hover) {
  background: var(--scrollbar-thumb-hover);
}

:deep(.markdown-body pre::-webkit-scrollbar-corner) {
  background: var(--scrollbar-corner);
}

:deep(.markdown-body pre code) {
  background: transparent;
  color: inherit;
  border: none;
  padding: 0;
}

:deep(.markdown-body table) {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 2rem;
  font-size: calc(0.9rem * var(--ui-scale));
}

:deep(.markdown-body th) {
  background: var(--bg-card);
  border: 1px solid var(--border-main);
  padding: 0.75rem;
  font-weight: 800;
  text-align: left;
}

:deep(.markdown-body td) {
  border: 1px solid var(--border-main);
  padding: 0.75rem;
}

:deep(.markdown-body ul) {
  list-style-type: disc;
  padding-left: 2rem;
  margin-bottom: 1.5rem;
}

:deep(.markdown-body ol) {
  list-style-type: decimal;
  padding-left: 2rem;
  margin-bottom: 1.5rem;
}

:deep(.markdown-body li) {
  margin-bottom: 0.5rem;
  display: list-item;
}

:deep(.markdown-body blockquote) {
  border-left: 4px solid var(--accent);
  background: var(--bg-card);
  padding: 1rem 1.5rem;
  margin: 1.5rem 0;
  border-radius: 0 8px 8px 0;
  font-style: italic;
  opacity: 0.8;
}

:deep(.markdown-body a) {
  color: var(--accent);
  text-decoration: underline;
  font-weight: 700;
}

:deep(.markdown-body a:hover) {
  opacity: 0.8;
}

:deep(.markdown-body hr) {
  border: 0;
  border-top: 1px solid var(--border-main);
  margin: 2rem 0;
}

:deep(.help-search-hit) {
  border-radius: 0.4rem;
  background: color-mix(in srgb, var(--accent) 22%, transparent);
  color: inherit;
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 24%, transparent);
  padding: 0.03em 0.12em;
}

:deep(.help-search-hit.is-active) {
  background: color-mix(in srgb, var(--accent) 42%, white 8%);
  box-shadow:
    inset 0 0 0 1px color-mix(in srgb, var(--accent) 45%, transparent),
    0 0 0 3px color-mix(in srgb, var(--accent) 16%, transparent);
}
</style>
