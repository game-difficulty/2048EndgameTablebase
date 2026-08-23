from __future__ import annotations


MINIGAMES: tuple[tuple[str, str], ...] = (
    ("design-master-1", "Design Master1"),
    ("mystery-merge-1", "Mystery Merge1"),
    ("column-chaos", "Column Chaos"),
    ("gravity-twist-1", "Gravity Twist1"),
    ("blitzkrieg", "Blitzkrieg"),
    ("tricky-tiles", "Tricky Tiles"),
    ("design-master-2", "Design Master2"),
    ("shape-shifter", "Shape Shifter"),
    ("ferris-wheel", "Ferris Wheel"),
    ("gravity-twist-2", "Gravity Twist2"),
    ("design-master-3", "Design Master3"),
    ("mystery-merge-2", "Mystery Merge2"),
    ("ice-age", "Ice Age"),
    ("isolated-island", "Isolated Island"),
    ("design-master-4", "Design Master4"),
    ("endless-factorization", "Endless Factorization"),
    ("endless-explosions", "Endless Explosions"),
    ("endless-giftbox", "Endless Giftbox"),
    ("endless-hybrid", "Endless Hybrid"),
    ("endless-airraid", "Endless AirRaid"),
)

MINIGAME_BY_ID = dict(MINIGAMES)


def minigame_catalog() -> list[dict[str, str]]:
    return [{"id": game_id, "title": title} for game_id, title in MINIGAMES]
