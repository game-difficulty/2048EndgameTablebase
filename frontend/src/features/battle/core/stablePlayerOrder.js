import { battleActorRenderKey } from './battleActor.js';

export function createStablePlayerOrder() {
  let roundId = null;
  const positions = new Map();

  return (room) => {
    const nextRoundId = `${room?.room_id || ''}:${room?.round?.round_id || ''}`;
    if (nextRoundId !== roundId) {
      roundId = nextRoundId;
      positions.clear();
    }
    const members = new Map((room?.members || []).map((member) => [battleActorRenderKey(member), member]));
    const rows = (room?.results || []).map((result) => {
      const member = members.get(battleActorRenderKey(result)) || {};
      return { ...member, ...result };
    });
    const newcomers = rows.filter((row) => !positions.has(battleActorRenderKey(row)));
    newcomers.sort((left, right) => {
      const leftSeat = Number(left.seat_index);
      const rightSeat = Number(right.seat_index);
      const a = Number.isInteger(leftSeat) && leftSeat >= 0 ? leftSeat : Infinity;
      const b = Number.isInteger(rightSeat) && rightSeat >= 0 ? rightSeat : Infinity;
      return a - b || battleActorRenderKey(left).localeCompare(battleActorRenderKey(right));
    });
    for (const row of newcomers) positions.set(battleActorRenderKey(row), positions.size);
    return rows.sort((left, right) => positions.get(battleActorRenderKey(left)) - positions.get(battleActorRenderKey(right)));
  };
}
