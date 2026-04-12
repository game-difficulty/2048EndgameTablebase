import { nextTick, onMounted, onUnmounted, ref, watch } from 'vue';

import { createWsClient } from '../../../services/ws/createWsClient';

export function useHelpSession(activeRef) {
  const htmlContent = ref('');
  const toc = ref([]);
  const loading = ref(true);
  const contentArea = ref(null);
  const articleRef = ref(null);
  const searchInput = ref(null);
  const searchOpen = ref(false);
  const searchQuery = ref('');
  const searchMatchCount = ref(0);
  const activeSearchIndex = ref(0);

  let highlightedMatches = [];
  let searchRefreshTimer = null;

  const typesetMath = async () => {
    await nextTick();
    window.setTimeout(() => {
      if (window.MathJax && window.MathJax.typeset) {
        window.MathJax.typeset();
      }
    }, 100);
  };

  let client = null;

  const escapeRegExp = (value) =>
    value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  const clearSearchRefreshTimer = () => {
    if (searchRefreshTimer) {
      window.clearTimeout(searchRefreshTimer);
      searchRefreshTimer = null;
    }
  };

  const clearHighlights = () => {
    const article = articleRef.value;
    if (!article) {
      highlightedMatches = [];
      searchMatchCount.value = 0;
      activeSearchIndex.value = 0;
      return;
    }

    const marks = article.querySelectorAll('.help-search-hit');
    marks.forEach((mark) => {
      const parent = mark.parentNode;
      if (!parent) {
        return;
      }
      parent.replaceChild(document.createTextNode(mark.textContent || ''), mark);
      parent.normalize();
    });

    highlightedMatches = [];
    searchMatchCount.value = 0;
    activeSearchIndex.value = 0;
  };

  const scrollToMatch = (element) => {
    if (!element) {
      return;
    }
    element.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
      inline: 'nearest',
    });
  };

  const setActiveMatch = (index, shouldScroll = true) => {
    if (!highlightedMatches.length) {
      activeSearchIndex.value = 0;
      return;
    }

    const normalizedIndex =
      ((index % highlightedMatches.length) + highlightedMatches.length) %
      highlightedMatches.length;

    highlightedMatches.forEach((element, currentIndex) => {
      element.classList.toggle('is-active', currentIndex === normalizedIndex);
    });

    activeSearchIndex.value = normalizedIndex;
    if (shouldScroll) {
      scrollToMatch(highlightedMatches[normalizedIndex]);
    }
  };

  const applySearchHighlights = () => {
    const article = articleRef.value;
    clearHighlights();

    if (!article) {
      return;
    }

    const query = searchQuery.value.trim();
    if (!query) {
      return;
    }

    const matcher = new RegExp(escapeRegExp(query), 'gi');
    const walker = document.createTreeWalker(
      article,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          const value = node.nodeValue || '';
          if (!value.trim()) {
            return NodeFilter.FILTER_REJECT;
          }

          const parent = node.parentElement;
          if (!parent || parent.closest('.help-search-hit, script, style, textarea')) {
            return NodeFilter.FILTER_REJECT;
          }

          return NodeFilter.FILTER_ACCEPT;
        },
      }
    );

    const textNodes = [];
    let currentNode = walker.nextNode();
    while (currentNode) {
      textNodes.push(currentNode);
      currentNode = walker.nextNode();
    }

    const nextMatches = [];

    textNodes.forEach((node) => {
      const text = node.nodeValue || '';
      matcher.lastIndex = 0;
      if (!matcher.test(text)) {
        return;
      }

      matcher.lastIndex = 0;
      const fragment = document.createDocumentFragment();
      let lastIndex = 0;
      let match = matcher.exec(text);

      while (match) {
        const matchText = match[0];
        const startIndex = match.index;
        if (startIndex > lastIndex) {
          fragment.appendChild(
            document.createTextNode(text.slice(lastIndex, startIndex))
          );
        }

        const mark = document.createElement('mark');
        mark.className = 'help-search-hit';
        mark.textContent = matchText;
        fragment.appendChild(mark);
        nextMatches.push(mark);

        lastIndex = startIndex + matchText.length;
        match = matcher.exec(text);
      }

      if (lastIndex < text.length) {
        fragment.appendChild(document.createTextNode(text.slice(lastIndex)));
      }

      node.parentNode?.replaceChild(fragment, node);
    });

    highlightedMatches = nextMatches;
    searchMatchCount.value = nextMatches.length;
    if (nextMatches.length > 0) {
      setActiveMatch(0, false);
    }
  };

  const scheduleSearchRefresh = () => {
    clearSearchRefreshTimer();
    searchRefreshTimer = window.setTimeout(() => {
      if (searchOpen.value && searchQuery.value.trim()) {
        applySearchHighlights();
      }
    }, 180);
  };

  const openSearch = async () => {
    searchOpen.value = true;
    await nextTick();
    searchInput.value?.focus();
    searchInput.value?.select?.();
    if (searchQuery.value.trim()) {
      applySearchHighlights();
    }
  };

  const closeSearch = () => {
    searchOpen.value = false;
    searchQuery.value = '';
    clearHighlights();
  };

  const focusSearch = async () => {
    if (!searchOpen.value) {
      await openSearch();
      return;
    }
    await nextTick();
    searchInput.value?.focus();
    searchInput.value?.select?.();
  };

  const nextMatch = () => {
    if (!highlightedMatches.length) {
      return;
    }
    setActiveMatch(activeSearchIndex.value + 1);
  };

  const previousMatch = () => {
    if (!highlightedMatches.length) {
      return;
    }
    setActiveMatch(activeSearchIndex.value - 1);
  };

  const handleGlobalKeydown = (event) => {
    if (!activeRef?.value) {
      return;
    }

    const isFindShortcut = (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'f';
    if (isFindShortcut) {
      event.preventDefault();
      focusSearch();
      return;
    }

    if (event.key === 'F3') {
      event.preventDefault();
      if (!searchOpen.value) {
        openSearch();
        return;
      }
      if (event.shiftKey) {
        previousMatch();
      } else {
        nextMatch();
      }
      return;
    }

    if (!searchOpen.value) {
      return;
    }

    if (event.key === 'Escape') {
      event.preventDefault();
      closeSearch();
      return;
    }

    if (event.key === 'Enter') {
      event.preventDefault();
      if (event.shiftKey) {
        previousMatch();
      } else {
        nextMatch();
      }
    }
  };

  const fetchHelp = () => {
    loading.value = true;
    client?.send('GET_HELP');
  };

  const connect = () => {
    if (client) {
      return;
    }
    client = createWsClient({
      clientId: Date.now().toString(),
      onOpen: () => {
        if (activeRef?.value) {
          fetchHelp();
        }
      },
      onMessage: async (data) => {
        if (data.type !== 'HELP_DATA') {
          return;
        }
        htmlContent.value = data.payload.html;
        toc.value = data.payload.toc;
        loading.value = false;
        await typesetMath();
        scheduleSearchRefresh();
      },
    });
    client.connect();
  };

  const disconnect = () => {
    client?.disconnect();
    client = null;
  };

  const scrollTo = (id) => {
    const element = document.getElementById(id);
    if (!element || !contentArea.value) {
      return;
    }
    contentArea.value.scrollTo({
      top: element.offsetTop - 20,
      behavior: 'smooth',
    });
  };

  onUnmounted(() => {
    clearSearchRefreshTimer();
    clearHighlights();
    disconnect();
  });

  onMounted(() => {
    window.addEventListener('keydown', handleGlobalKeydown, true);
  });

  onUnmounted(() => {
    window.removeEventListener('keydown', handleGlobalKeydown, true);
  });

  watch(
    activeRef,
    (isActive) => {
      if (isActive) {
        connect();
        if (!htmlContent.value) {
          fetchHelp();
        }
      } else {
        disconnect();
      }
    },
    { immediate: true }
  );

  watch(searchQuery, () => {
    if (!searchOpen.value) {
      return;
    }
    applySearchHighlights();
  });

  return {
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
  };
}
