import { ref, watch, onBeforeUnmount } from "vue";
import { sessionHeaders } from "./api";
export function useNotifications(user) {
  const status = ref({ latest: 0, unread: 0, enabled: true }),
    connected = ref(false);
  let controller,
    timer,
    generation = 0;
  function clear() {
    generation++;
    controller?.abort();
    clearTimeout(timer);
    connected.value = false;
  }
  async function connect(ticket) {
    controller = new AbortController();
    try {
      const response = await fetch("/api/forum/v1/notification-stream", {
        credentials: "include",
        headers: { ...sessionHeaders(), Accept: "text/event-stream" },
        signal: controller.signal,
      });
      if (response.status === 401) {
        window.dispatchEvent(new Event("forum-auth-expired"));
        return;
      }
      if (!response.ok || !response.body) throw Error("stream unavailable");
      connected.value = true;
      const reader = response.body.getReader(),
        decoder = new TextDecoder();
      let buffer = "";
      while (ticket === generation) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        if (buffer.length > 65536) throw Error("stream overflow");
        let end;
        while ((end = buffer.indexOf("\n\n")) !== -1) {
          const packet = buffer.slice(0, end);
          buffer = buffer.slice(end + 2);
          const line = packet.split("\n").find((l) => l.startsWith("data: "));
          if (line && ticket === generation)
            status.value = JSON.parse(line.slice(6));
        }
      }
    } catch {
      /* The durable inbox is re-read on reconnect. */
    } finally {
      if (ticket === generation) {
        connected.value = false;
        timer = setTimeout(() => connect(ticket), 3000);
      }
    }
  }
  watch(
    () => user.value?.id,
    (id) => {
      clear();
      status.value = { latest: 0, unread: 0, enabled: true };
      if (id) connect(generation);
    },
    { immediate: true },
  );
  onBeforeUnmount(clear);
  return { status, connected };
}
