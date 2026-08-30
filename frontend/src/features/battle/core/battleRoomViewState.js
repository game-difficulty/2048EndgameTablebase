function roundId(room) {
  return String(room?.round?.round_id || '');
}

function isCompleted(room) {
  return Boolean(roundId(room) && room?.round?.status === 'completed');
}

export function createBattleRoomViewState() {
  let roomId = '';
  let activeRoundId = '';
  let heldRoundId = '';
  let returnedRoundId = '';
  let resultRoundId = '';

  const snapshot = () => ({ heldRoundId, resultRoundId });
  const reset = () => {
    roomId = '';
    activeRoundId = '';
    heldRoundId = '';
    returnedRoundId = '';
    resultRoundId = '';
  };

  const apply = (previousRoom, nextRoom) => {
    if (!nextRoom) {
      reset();
      return snapshot();
    }
    const nextRoomId = String(nextRoom.room_id || '');
    if (roomId && nextRoomId !== roomId) reset();
    roomId = nextRoomId;

    const nextRoundId = roundId(nextRoom);
    const previousWasRunning = (
      previousRoom?.status === 'running'
      && roundId(previousRoom) === nextRoundId
    );
    if (nextRoom.status === 'running' && nextRoundId) {
      activeRoundId = nextRoundId;
      heldRoundId = '';
      returnedRoundId = '';
      resultRoundId = '';
      return snapshot();
    }
    if (isCompleted(nextRoom)) {
      if (
        returnedRoundId !== nextRoundId
        && (activeRoundId === nextRoundId || previousWasRunning)
      ) {
        heldRoundId = nextRoundId;
      }
      return snapshot();
    }
    if (nextRoundId !== heldRoundId) {
      heldRoundId = '';
      resultRoundId = '';
    }
    return snapshot();
  };

  const returnToLobby = (room) => {
    const currentRoundId = roundId(room);
    if (currentRoundId) returnedRoundId = currentRoundId;
    heldRoundId = '';
    resultRoundId = '';
    return snapshot();
  };

  const openResults = (room) => {
    const currentRoundId = roundId(room);
    if (currentRoundId && currentRoundId === heldRoundId) {
      resultRoundId = currentRoundId;
    }
    return snapshot();
  };

  const closeResults = () => {
    resultRoundId = '';
    return snapshot();
  };

  return { apply, returnToLobby, openResults, closeResults };
}
