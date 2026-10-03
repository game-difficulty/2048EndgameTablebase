// Group only loaded, equally visible notices. Keep every floor link in the group.
export function groupNotifications(items) {
  const groups = new Map();
  for (const item of items) {
    const key = item.topic_id
      ? [
          item.topic_id,
          item.kind,
          !!item.read_at,
          item.available,
          String(item.created_at).slice(0, 10),
        ].join(":")
      : `system:${item.id}`;
    if (!groups.has(key)) groups.set(key, { ...item, notices: [] });
    groups.get(key).notices.push(item);
  }
  return [...groups.values()];
}
