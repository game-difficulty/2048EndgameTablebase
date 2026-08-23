from __future__ import annotations

from dataclasses import dataclass


UINT32_MASK = 0xFFFFFFFF


def _rotl(value: int, shift: int) -> int:
    value &= UINT32_MASK
    return ((value << shift) | (value >> (32 - shift))) & UINT32_MASK


@dataclass
class Xoshiro128StarStar:
    state: list[int]

    @classmethod
    def from_seed_hex(cls, seed_hex: str) -> "Xoshiro128StarStar":
        normalized = str(seed_hex or "").strip().lower()
        if len(normalized) != 32:
            raise ValueError("Ranked seed must contain 128 bits.")
        try:
            state = [int(normalized[index:index + 8], 16) for index in range(0, 32, 8)]
        except ValueError as exc:
            raise ValueError("Ranked seed is not hexadecimal.") from exc
        if any(value == 0 for value in state):
            raise ValueError("Ranked seed words must be non-zero.")
        return cls(state)

    def next_u32(self) -> int:
        s0, s1, s2, s3 = self.state
        result = (_rotl((s1 * 5) & UINT32_MASK, 7) * 9) & UINT32_MASK
        temporary = (s1 << 9) & UINT32_MASK
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= temporary
        s3 = _rotl(s3, 11)
        self.state[:] = [
            s0 & UINT32_MASK,
            s1 & UINT32_MASK,
            s2 & UINT32_MASK,
            s3 & UINT32_MASK,
        ]
        return result

    def next_float(self) -> float:
        return self.next_u32() / 4294967296.0

    def choose_index(self, count: int) -> int:
        if count <= 0:
            raise ValueError("Cannot select from an empty collection.")
        return self.next_u32() % count
