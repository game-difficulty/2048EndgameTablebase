const prefix = "/api/forum/v1";
const devUser = import.meta.env.DEV
  ? String(import.meta.env.VITE_FORUM_DEV_USER || "")
  : "";

export async function api(path, { method = "GET", body, key, signal } = {}) {
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (key) headers["Idempotency-Key"] = key;
  if (devUser) headers["X-Forum-Dev-User"] = devUser;
  const response = await fetch(prefix + path, {
    method,
    headers,
    credentials: "include",
    signal,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    let message = data.detail?.message;
    if (Array.isArray(data.detail))
      message = "内容格式不正确，请检查标题、正文和棋盘。";
    const error = new Error(message || "请求失败，请稍后重试。");
    error.status = response.status;
    error.code = data.detail?.code;
    if (response.status === 401)
      window.dispatchEvent(new Event("forum-auth-expired"));
    throw error;
  }
  return data;
}

// Reuse the key for uncertain retries, but not after the user changes the payload.
export function submission() {
  let previous = "",
    key = "";
  const next = (payload) => {
    const fingerprint = JSON.stringify(payload);
    if (fingerprint !== previous) {
      previous = fingerprint;
      key = crypto.randomUUID();
    }
    return key;
  };
  next.reset = () => {
    previous = "";
    key = "";
  };
  return next;
}
