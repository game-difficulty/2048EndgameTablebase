"""Shared table goal identity. Legacy numeric targets keep their existing names."""
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class GoalSpec:
    kind: str
    value: int

    @classmethod
    def parse(cls, value, *, rank=False):
        if isinstance(value, cls):
            return value
        text = str(value).strip()
        if text.startswith("sum-"):
            number = int(text[4:])
            if number < 4 or number >= 16384 or number % 2:
                raise ValueError("Sum target must be an even integer from 4 to 16382")
            return cls("sum", number)
        number = int(text)
        if rank:
            if not 1 <= number <= 14:
                raise ValueError("Tile exponent must be in [1, 14]")
            number = 1 << number
        if number < 2 or number > 16384 or number & (number - 1):
            raise ValueError("Tile target must be a power of two from 2 to 16384")
        return cls("tile", number)

    @classmethod
    def from_prefix(cls, prefix):
        token = Path(str(prefix).rstrip("_")).name.rsplit("_", 1)[-1]
        return cls.parse(token)

    @property
    def token(self):
        return f"sum-{self.value}" if self.kind == "sum" else str(self.value)

    @property
    def encoding_rank(self):
        return self.value.bit_length() - 1

    @property
    def sum_target(self):
        return self.value if self.kind == "sum" else 0

    def build_range(self, legacy_sum, extra_steps, *, start_offset=0):
        if self.kind == "tile":
            return self.value // 2 + extra_steps, self.value // 2 - legacy_sum % self.value // 2
        initial_sum = legacy_sum + 2 * start_offset
        base = initial_sum - initial_sum % 16384
        boundary = base + self.value - 2
        if initial_sum >= boundary:
            raise ValueError("Initial layer already reaches this sum target; choose a larger target")
        first = (boundary - legacy_sum) // 2
        return first + 2, first - 1

    def reached(self, board_encoded):
        value = int(board_encoded)
        total = sum((1 << ((value >> shift) & 15)) if ((value >> shift) & 15) else 0
                    for shift in range(0, 64, 4))
        return self.kind == "sum" and total % 16384 >= self.value - 2

    def validate_algorithm(self, algorithm):
        if self.kind == "sum" and algorithm in ("ad", "exad"):
            raise ValueError("Sum targets support Classic, EX and supported BC patterns; AD/EXAD are unavailable")

    def save_metadata(self, prefix, legacy_sum, extra_steps, start_offset, algorithm):
        if self.kind != "sum":
            return
        self.validate_algorithm(algorithm)
        if self.from_prefix(prefix) != self:
            raise ValueError("Output prefix must contain the canonical goal token: " + self.token)
        steps, _ = self.build_range(legacy_sum, extra_steps, start_offset=start_offset)
        data = dict(version=1, kind=self.kind, value=self.value,
                    semantics="post_move_mod16384_target_minus_2", encoding_rank=self.encoding_rank,
                    layer_zero_sum=legacy_sum, layer_offset=start_offset,
                    terminal_layers=[steps - 2, steps - 1], algorithm=algorithm)
        path = Path(str(prefix) + "goal.json")
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != data:
            raise ValueError("Existing table goal metadata differs; choose a separate output prefix")
        if not path.exists():
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)


def sum_target_from_prefix(prefix):
    token = Path(str(prefix).rstrip("_")).name.rsplit("_", 1)[-1]
    return GoalSpec.parse(token).sum_target if token.startswith("sum-") else 0


def available_target_tokens():
    from Config import SingletonConfig
    tokens = {str(1 << rank) for rank in range(6, 15)}
    for targets in SingletonConfig.get_available_pattern_targets().values():
        tokens.update(str(target) for target in targets)
    def key(token):
        try:
            goal = GoalSpec.parse(token)
            return (goal.kind == "sum", goal.value)
        except ValueError:
            return (2, token)
    return sorted(tokens, key=key)
