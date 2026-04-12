from __future__ import annotations

from minigames.MinigameMover import MinigameBoardMover, MinigameBoardMover_mxn


def create_mover(shape: tuple[int, int]):
    return (
        MinigameBoardMover()
        if shape == (4, 4)
        else MinigameBoardMover_mxn(shape=shape)
    )


__all__ = ["MinigameBoardMover", "MinigameBoardMover_mxn", "create_mover"]
