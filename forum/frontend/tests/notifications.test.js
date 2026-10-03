import test from "node:test";
import assert from "node:assert/strict";
import { groupNotifications } from "../src/notificationGroups.js";
test("notification grouping retains anchors and separates unread, visibility, day and type", () => {
  const base = {
    topic_id: 2,
    kind: "reply",
    read_at: null,
    available: true,
    created_at: "2026-10-03T10:00:00Z",
  };
  const items = [
    { ...base, id: 7, path: "/t/2#p-7" },
    { ...base, id: 6, path: "/t/2#p-6" },
    { ...base, id: 5, available: false },
    { ...base, id: 4, read_at: "now" },
    { ...base, id: 3, kind: "mention" },
    { ...base, id: 2, created_at: "2026-10-02T10:00:00Z" },
    { ...base, id: 1, topic_id: null },
  ];
  const result = groupNotifications(items);
  assert.equal(result.length, 6);
  assert.deepEqual(
    result[0].notices.map((n) => n.path),
    ["/t/2#p-7", "/t/2#p-6"],
  );
});
