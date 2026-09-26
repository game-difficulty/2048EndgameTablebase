export function isRoomEndedEvent(message, roomId) {
  return message?.type === 'room_ended'
    && (!message.room_id || message.room_id === roomId);
}
