const DEFAULT_GUIDE_INDEX = `${import.meta.env?.BASE_URL || '/'}guides/index.json`;

function resolveUrl(value, baseUrl) {
  const pageUrl = globalThis.location?.href || 'http://localhost/';
  return new URL(value, new URL(baseUrl, pageUrl)).toString();
}

async function fetchJson(url, fetchImpl, cache = 'force-cache') {
  const response = await fetchImpl(url, { cache });
  if (!response.ok) {
    throw new Error(`Guide request failed with HTTP ${response.status}.`);
  }
  return {
    data: await response.json(),
    responseUrl: response.url || resolveUrl(url, globalThis.location?.href || 'http://localhost/'),
  };
}

export async function loadGuideIndex(fetchImpl = globalThis.fetch, indexUrl = DEFAULT_GUIDE_INDEX) {
  if (typeof fetchImpl !== 'function') {
    throw new Error('Fetch is unavailable.');
  }

  const { data, responseUrl } = await fetchJson(indexUrl, fetchImpl, 'no-cache');
  const documents = Array.isArray(data?.documents) ? data.documents : [];

  return documents
    .filter((entry) => entry?.id && entry?.source && entry?.title)
    .map((entry) => ({
      ...entry,
      documentUrl: resolveUrl(entry.source, responseUrl),
    }));
}

export async function loadGuideDocument(entry, fetchImpl = globalThis.fetch) {
  if (!entry?.documentUrl) {
    throw new Error('Guide document URL is missing.');
  }

  const { data, responseUrl } = await fetchJson(entry.documentUrl, fetchImpl);
  if (!data?.id || !Array.isArray(data?.blocks) || !Array.isArray(data?.toc)) {
    throw new Error('Guide document has an unsupported structure.');
  }

  return {
    ...data,
    blocks: data.blocks.map((block) => (
      block?.type === 'figure' && block?.src
        ? { ...block, src: resolveUrl(block.src, responseUrl) }
        : block
    )),
  };
}
