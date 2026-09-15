"""Page-session scoped, single-use Default query authorization."""
import secrets
import time


class TrainerDefaultLookup:
    def __init__(self):
        self.free_at = {}
        self.pending = None

    def issue(self, pattern, board, *, switched):
        ticket = secrets.token_hex(16)
        self.pending = (ticket, pattern, int(board), bool(switched), time.monotonic())
        return ticket

    def consume(self, ticket, pattern, board):
        now = time.monotonic()
        pending = self.pending
        if not pending or pending[:3] != (ticket, pattern, int(board)) or now - pending[4] > 120:
            return False, False
        self.pending = None
        self.free_at = {key: at for key, at in self.free_at.items() if now - at < 600}
        free = pending[3] and pattern not in self.free_at
        if free:
            self.free_at[pattern] = now
        return True, free


def default_lookup_for(session):
    policy = getattr(session, "trainer_default_lookup", None)
    if policy is None:
        policy = session.trainer_default_lookup = TrainerDefaultLookup()
    return policy
