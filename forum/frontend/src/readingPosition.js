import { ref, watch, nextTick, onBeforeUnmount } from "vue";
import { api } from "./api";

export function useReadingPosition(topicId, userId, data) {
  const position = ref(null);
  let context, observer, timer;
  async function send(c) {
    clearTimeout(timer);
    if (context !== c || !c?.pending || c.pending.number <= c.saved) return;
    const point = c.pending;
    try {
      await api(`/topics/${c.topic}/reading`, {
        method: "PUT",
        body: { post_id: point.id },
      });
      c.saved = Math.max(c.saved, point.number);
    } catch {
      /* Reading can retry on the next visible floor; it never blocks a reply. */
    }
  }
  watch(
    [topicId, userId],
    async () => {
      clearTimeout(timer);
      observer?.disconnect();
      position.value = null;
      context = null;
      if (!userId()) return;
      const c = (context = { topic: topicId(), saved: 0, pending: null });
      try {
        const result = await api(`/topics/${c.topic}/reading`);
        if (context === c) {
          position.value = result.position;
          c.saved = Math.max(c.saved, result.position?.post_number || 0);
        }
      } catch {
        /* public reading remains usable */
      }
    },
    { immediate: true },
  );
  watch(
    () => data.value?.posts,
    async () => {
      observer?.disconnect();
      await nextTick();
      const c = context;
      if (!c || !window.IntersectionObserver) return;
      observer = new IntersectionObserver(
        (entries) => {
          if (context !== c || document.visibilityState !== "visible") return;
          for (const entry of entries)
            if (entry.isIntersecting) {
              const number = Number(entry.target.dataset.number);
              if (number > (c.pending?.number || c.saved))
                c.pending = { number, id: Number(entry.target.dataset.post) };
            }
          if (c.pending && !timer)
            timer = setTimeout(() => {
              timer = null;
              send(c);
            }, 3000);
        },
        { threshold: 1 },
      );
      document
        .querySelectorAll(".read-marker")
        .forEach((el) => observer.observe(el));
    },
    { flush: "post" },
  );
  function flush() {
    return send(context);
  }
  onBeforeUnmount(() => {
    observer?.disconnect();
    clearTimeout(timer);
  });
  return { position, flush };
}
