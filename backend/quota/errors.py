from __future__ import annotations


class InsufficientTokens(RuntimeError):
    def __init__(self, *, required_units: int, balance_units: int):
        self.required_units = max(0, int(required_units))
        self.balance_units = max(0, int(balance_units))
        super().__init__("Insufficient token balance.")

    @property
    def payload(self) -> dict:
        return {
            "code": "INSUFFICIENT_TOKENS",
            "message": "Insufficient token balance.",
            "required_tokens": self.required_units / 1000,
            "balance_tokens": self.balance_units / 1000,
        }
