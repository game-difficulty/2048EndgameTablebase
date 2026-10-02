const prefix = "/api/forum/v1";
const devUser = import.meta.env.DEV
  ? String(import.meta.env.VITE_FORUM_DEV_USER || "")
  : "";

export function sessionHeaders() {
  return devUser ? { "X-Forum-Dev-User": devUser } : {};
}

export async function uploadAsset(file, kind) {
  const response = await fetch(prefix + "/media?kind=" + kind, {
    method: "POST",
    credentials: "include",
    headers: {
      ...sessionHeaders(),
      "Content-Type": "application/octet-stream",
    },
    body: file,
  });
  const data = await response.json();
  if (!response.ok)
    throw new Error(data.detail?.message || "上传失败，请重试。");
  return data;
}

export async function mediaBlob(id, signal) {
  const response = await fetch(prefix + "/media/" + id, {
    credentials: "include",
    headers: sessionHeaders(),
    signal,
  });
  if (!response.ok) throw new Error("图片已移除或不可见。");
  return response.blob();
}

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
