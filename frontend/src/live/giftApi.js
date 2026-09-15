export async function giftApi(path, body) {
  const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(`/api/live/gifts/${path}`, {
      credentials: 'same-origin', cache: 'no-store', signal: controller.signal,
      ...(body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
    });
    const data = await response.json();
    if (!response.ok) throw Object.assign(new Error('gift_failed'), { status: response.status, detail: data.detail });
    return data;
  } finally { clearTimeout(timer); }
}

export async function sendGift(request, recovering = false) {
  if (recovering) {
    try { return await giftApi(`orders/${request.request_id}`); }
    catch (error) { if (error.status !== 404) throw Object.assign(new Error('delivery_unconfirmed'), { cause: error }); }
  }
  // A transport failure never creates a second purchase identity.
  try { return await giftApi('send', request); }
  catch (error) {
    if (error.status && error.status < 500) throw error;
    try { return await giftApi('send', request); }
    catch (retryError) {
      try { return await giftApi(`orders/${request.request_id}`); }
      catch { throw Object.assign(new Error('delivery_unconfirmed'), { cause: retryError }); }
    }
  }
}
