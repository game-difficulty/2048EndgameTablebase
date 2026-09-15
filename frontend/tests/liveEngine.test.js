import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { applyLiveStep } from "../src/live/liveEngine.js";
import { simulateMove } from "../src/features/gamer/engine/classicMove.js";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/live.json", import.meta.url)),
);
test("browser reconstructs Python broadcast steps, score and milestones", () => {
  let run = fixture.initial;
  for (const packet of fixture.packets)
    run = applyLiveStep(run, Uint8Array.from(packet).buffer).run;
  for (const key of ["board", "score", "seq", "elapsed_ms", "nodes"])
    assert.deepEqual(run[key], fixture.final[key]);
});
test("duplicates and dropped steps require a snapshot", () => {
  const packet = Uint8Array.from(fixture.packets[0]).buffer;
  const { run } = applyLiveStep(fixture.initial, packet);
  assert.throws(() => applyLiveStep(run, packet), /snapshot_required/);
  assert.throws(
    () =>
      applyLiveStep(
        fixture.initial,
        Uint8Array.from(fixture.packets[1]).buffer,
      ),
    /snapshot_required/,
  );
});
test("shared Gamer movement handles 65K with correct score and animation metadata", () => {
  const result = simulateMove([32768, 32768, ...Array(14).fill(0)], "left");
  assert.equal(result.board[0], 65536);
  assert.equal(result.scoreDelta, 65536);
  assert.equal(result.popPositions[0], 1);
  assert.equal(result.slideDistances[1], 1);
});
