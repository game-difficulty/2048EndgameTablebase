import { ref, watch, onBeforeUnmount } from "vue";
import { api } from "./api";

// Each navigation owns its revision and serialized write queue. Old responses cannot change new editors.
export function useReplyDraft(topicId, userId, text, replyTo) {
  const ready = ref(false),
    status = ref(""),
    conflict = ref(null);
  let context,
    timer,
    restoring = false;
  const snapshot = () => ({
    text: text.value,
    reply_to: replyTo.value?.id || null,
  });
  function local(c, value) {
    try {
      localStorage.setItem(
        c.key,
        JSON.stringify({ ...value, revision: c.revision }),
      );
    } catch {
      status.value = "浏览器无法保留本地草稿，请及时保存到云端。";
    }
  }
  async function save() {
    clearTimeout(timer);
    const c = context;
    if (!c || !ready.value || c.blocked) return;
    const value = snapshot();
    local(c, value);
    c.queue = c.queue.then(async () => {
      if (context !== c || c.blocked || JSON.stringify(value) === c.saved)
        return;
      try {
        const r = await api(`/topics/${c.topic}/reply-draft`, {
          method: "PUT",
          body: { ...value, revision: c.revision },
        });
        c.revision = r.revision;
        c.saved = JSON.stringify(value);
        if (context === c) {
          local(c, snapshot());
          status.value = "回复草稿已保存";
        }
      } catch (e) {
        if (e.code === "REVISION_CONFLICT") {
          c.blocked = true;
          try {
            const r = await api(`/topics/${c.topic}/reply-draft`);
            if (context === c) conflict.value = r.draft;
          } catch {
            /* local text stays intact */
          }
        }
        if (context === c) status.value = e.message + " 本地文本已保留。";
      }
    });
    await c.queue;
  }
  async function restore() {
    clearTimeout(timer);
    ready.value = false;
    conflict.value = null;
    status.value = "";
    restoring = true;
    text.value = "";
    replyTo.value = null;
    restoring = false;
    context = null;
    if (!userId() || !topicId()) return;
    const c = (context = {
      topic: topicId(),
      key: `forum.reply.${userId()}.${topicId()}`,
      revision: 0,
      queue: Promise.resolve(),
      saved: "",
      blocked: false,
    });
    let cached;
    try {
      cached = JSON.parse(localStorage.getItem(c.key));
    } catch {
      /* optional local cache */
    }
    try {
      const { draft } = await api(`/topics/${c.topic}/reply-draft`);
      if (context !== c) return;
      c.revision = draft?.revision || 0;
      c.saved = JSON.stringify({
        text: draft?.text || "",
        reply_to: draft?.reply_to || null,
      });
      const value = cached || draft;
      restoring = true;
      text.value = value?.text || "";
      replyTo.value = value?.reply_to
        ? { id: value.reply_to, post_number: draft?.post_number || "?" }
        : null;
      restoring = false;
      if (
        cached &&
        cached.revision !== c.revision &&
        JSON.stringify({ text: cached.text, reply_to: cached.reply_to }) !==
          c.saved
      ) {
        c.blocked = true;
        conflict.value = draft || { text: "", revision: 0 };
        status.value = "本地与云端版本不同，请选择要保留的草稿。";
      } else status.value = text.value ? "已恢复回复草稿" : "";
      ready.value = true;
      if (!c.blocked && text.value) save();
    } catch (e) {
      if (context !== c) return;
      c.blocked = true;
      restoring = true;
      text.value = cached?.text || "";
      replyTo.value = cached?.reply_to
        ? { id: cached.reply_to, post_number: "?" }
        : null;
      restoring = false;
      ready.value = true;
      status.value = e.message + " 暂时只保存到本机，请重试同步。";
    }
  }
  async function resolve(useCloud) {
    if (!conflict.value || !context) return;
    const cloud = conflict.value;
    context.revision = cloud.revision;
    context.blocked = false;
    conflict.value = null;
    if (useCloud) {
      restoring = true;
      text.value = cloud.text;
      replyTo.value = cloud.reply_to
        ? { id: cloud.reply_to, post_number: cloud.post_number }
        : null;
      restoring = false;
    }
    context.saved = "";
    await save();
  }
  async function published() {
    await save();
    if (!context) return;
    restoring = true;
    text.value = "";
    replyTo.value = null;
    restoring = false;
    // A newer cloud draft is never erased after publishing an older local copy.
    if (context.blocked) {
      try {
        localStorage.removeItem(context.key);
      } catch {
        /* optional cache */
      }
      status.value = "已发布；云端存在另一份草稿，请刷新查看。";
      return;
    }
    await save();
  }
  watch([topicId, userId], restore, { immediate: true });
  watch(
    [text, () => replyTo.value?.id],
    () => {
      if (restoring || !ready.value || !context) return;
      local(context, snapshot());
      clearTimeout(timer);
      timer = setTimeout(save, 1200);
    },
    { flush: "sync" },
  );
  onBeforeUnmount(() => {
    clearTimeout(timer);
  });
  return { ready, status, conflict, save, resolve, published, restore };
}
