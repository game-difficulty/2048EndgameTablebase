"""Project-owned result semantics, independent of rooms and display protocols."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ResultPolicy:
    metric: str = 'score'
    race: bool = False

    def compare(self, yellow, white):
        y, w = (int(s.extra.get('result_value', s.score)) for s in (yellow, white))
        if yellow.outcome == 'surrendered':
            return y, w, 'white', 'yellow_surrendered'
        if white.outcome == 'surrendered':
            return y, w, 'yellow', 'white_surrendered'
        if self.race:
            yt, wt = yellow.outcome == 'target_reached', white.outcome == 'target_reached'
            if yt != wt:
                return y, w, 'yellow' if yt else 'white', 'race_target'
            if yt and wt:
                winner = 'yellow' if yellow.elapsed_ms < white.elapsed_ms else 'white' if white.elapsed_ms < yellow.elapsed_ms else 'draw'
                return y, w, winner, 'race_elapsed'
        return y, w, 'yellow' if y > w else 'white' if w > y else 'draw', self.metric
