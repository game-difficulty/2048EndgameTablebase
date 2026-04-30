from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from urllib.parse import quote


ASSET_BASE_PATH: Final[str] = "/minigames-assets"


@dataclass(frozen=True)
class MinigameDefinition:
    id: str
    title: str
    legacy_name: str
    module_key: str
    section: str
    cover_asset: str
    description: str
    difficulty_aware: bool = True
    supports_powerups: bool = True
    shape_mode: str = "fixed"
    hud_schema: tuple[str, ...] = ()
    view_flags: tuple[str, ...] = ()
    implemented: bool = False

    @property
    def cover_url(self) -> str:
        asset_path = self.cover_asset if "/" in self.cover_asset else f"covers/{self.cover_asset}"
        return f"{ASSET_BASE_PATH}/{quote(asset_path)}"


MINIGAME_REGISTRY: Final[list[MinigameDefinition]] = [
    MinigameDefinition(
        id="design-master-1",
        title="Design Master1",
        legacy_name="Design Master1",
        module_key="design_master",
        section="Classic Challenges",
        cover_asset="Design Master1.png",
        description="Match an exact target pattern on the board.",
        hud_schema=("patternText",),
        view_flags=("smallLabels",),
        implemented=True,
    ),
    MinigameDefinition(
        id="mystery-merge-1",
        title="Mystery Merge1",
        legacy_name="Mystery Merge1",
        module_key="mystery_merge",
        section="Classic Challenges",
        cover_asset="Mystery Merge1.png",
        description="Most tiles are concealed until key moments.",
        view_flags=("hiddenMask", "tileTextOverride"),
        hud_schema=("actionButton",),
        implemented=True,
    ),
    MinigameDefinition(
        id="column-chaos",
        title="Column Chaos",
        legacy_name="Column Chaos",
        module_key="column_chaos",
        section="Classic Challenges",
        cover_asset="Column Chaos.png",
        description="Survive random column swaps while building your score.",
        hud_schema=("remainingSteps",),
        implemented=True,
    ),
    MinigameDefinition(
        id="gravity-twist-1",
        title="Gravity Twist1",
        legacy_name="Gravity Twist1",
        module_key="gravity_twist",
        section="Classic Challenges",
        cover_asset="Gravity Twist1.png",
        description="Tiles keep falling under a strange gravity field.",
        implemented=True,
    ),
    MinigameDefinition(
        id="blitzkrieg",
        title="Blitzkrieg",
        legacy_name="Blitzkrieg",
        module_key="blitzkrieg",
        section="Classic Challenges",
        cover_asset="Blitzkrieg.png",
        description="Race the countdown and chain high-value merges for bonus time.",
        hud_schema=("countdown",),
        implemented=True,
    ),
    MinigameDefinition(
        id="tricky-tiles",
        title="Tricky Tiles",
        legacy_name="Tricky Tiles",
        module_key="tricky_tiles",
        section="Classic Challenges",
        cover_asset="Tricky Tiles.png",
        description="Every new tile tries to land somewhere inconvenient.",
        implemented=True,
    ),
    MinigameDefinition(
        id="design-master-2",
        title="Design Master2",
        legacy_name="Design Master2",
        module_key="design_master",
        section="Classic Challenges",
        cover_asset="Design Master2.png",
        description="A diagonal target pattern with strict spacing requirements.",
        hud_schema=("patternText",),
        view_flags=("smallLabels",),
        implemented=True,
    ),
    MinigameDefinition(
        id="shape-shifter",
        title="Shape Shifter",
        legacy_name="Shape Shifter",
        module_key="shape_shifter",
        section="Classic Challenges",
        cover_asset="Shape Shifter.png",
        description="The playable board shape changes from run to run.",
        shape_mode="dynamic",
        view_flags=("blockedMask",),
        implemented=True,
    ),
    MinigameDefinition(
        id="ferris-wheel",
        title="Ferris Wheel",
        legacy_name="Ferris Wheel",
        module_key="ferris_wheel",
        section="Classic Challenges",
        cover_asset="Ferris Wheel.png",
        description="A rotating board twists your plans after each move.",
        hud_schema=("remainingSteps",),
        implemented=True,
    ),
    MinigameDefinition(
        id="gravity-twist-2",
        title="Gravity Twist2",
        legacy_name="Gravity Twist2",
        module_key="gravity_twist",
        section="Classic Challenges",
        cover_asset="Gravity Twist2.png",
        description="Gravity not only drops tiles, it can merge them too.",
        implemented=True,
    ),
    MinigameDefinition(
        id="design-master-3",
        title="Design Master3",
        legacy_name="Design Master3",
        module_key="design_master",
        section="Extended Challenges",
        cover_asset="Design Master3.png",
        description="Sparse target anchors reward disciplined board control.",
        hud_schema=("patternText",),
        view_flags=("smallLabels",),
        implemented=True,
    ),
    MinigameDefinition(
        id="mystery-merge-2",
        title="Mystery Merge2",
        legacy_name="Mystery Merge2",
        module_key="mystery_merge",
        section="Extended Challenges",
        cover_asset="Mystery Merge2.png",
        description="Fresh tiles arrive hidden, and merges reveal only part of the truth.",
        view_flags=("hiddenMask", "tileTextOverride"),
        hud_schema=("actionButton",),
        implemented=True,
    ),
    MinigameDefinition(
        id="ice-age",
        title="Ice Age",
        legacy_name="Ice Age",
        module_key="ice_age",
        section="Extended Challenges",
        cover_asset="Ice Age.png",
        description="Tiles freeze in place if they stand still for too long.",
        view_flags=("tileOverlays", "coverSprites", "tileStyleVariant"),
        implemented=True,
    ),
    MinigameDefinition(
        id="isolated-island",
        title="Isolated Island",
        legacy_name="Isolated Island",
        module_key="isolated_island",
        section="Extended Challenges",
        cover_asset="Isolated Island.png",
        description="A fragmented board forces you to route merges around islands.",
        view_flags=("coverSprites", "tileStyleVariant"),
        implemented=True,
    ),
    MinigameDefinition(
        id="design-master-4",
        title="Design Master4",
        legacy_name="Design Master4",
        module_key="design_master",
        section="Extended Challenges",
        cover_asset="Design Master4.png",
        description="Fractional anchor weights create a demanding asymmetric target.",
        hud_schema=("patternText",),
        view_flags=("smallLabels",),
        implemented=True,
    ),
    MinigameDefinition(
        id="endless-factorization",
        title="Endless Factorization",
        legacy_name="Endless Factorization",
        module_key="endless_family",
        section="Endless Experiments",
        cover_asset="Endless Factorization.png",
        description="Endless mode with special factorization events and traps.",
        view_flags=("coverSprites", "tileTextOverride", "tileStyleVariant"),
        implemented=True,
    ),
    MinigameDefinition(
        id="endless-explosions",
        title="Endless Explosions",
        legacy_name="Endless Explosions",
        module_key="endless_family",
        section="Endless Experiments",
        cover_asset="Endless Explosions.png",
        description="Explosive tiles punctuate an endless score attack.",
        view_flags=("coverSprites",),
        implemented=True,
    ),
    MinigameDefinition(
        id="endless-giftbox",
        title="Endless Giftbox",
        legacy_name="Endless Giftbox",
        module_key="endless_family",
        section="Endless Experiments",
        cover_asset="Endless Giftbox.png",
        description="Special giftboxes keep the board unpredictable forever.",
        view_flags=("coverSprites",),
        implemented=True,
    ),
    MinigameDefinition(
        id="endless-hybrid",
        title="Endless Hybrid",
        legacy_name="Endless Hybrid",
        module_key="endless_family",
        section="Endless Experiments",
        cover_asset="Endless Hybrid.png",
        description="A chaotic mash-up of the endless rule variants.",
        view_flags=("coverSprites", "tileTextOverride", "tileStyleVariant"),
        implemented=True,
    ),
    MinigameDefinition(
        id="endless-airraid",
        title="Endless AirRaid",
        legacy_name="Endless AirRaid",
        module_key="endless_family",
        section="Endless Experiments",
        cover_asset="Endless AirRaid.png",
        description="Incoming strikes reshape the endless battlefield.",
        view_flags=("coverSprites", "blockedMask"),
        implemented=True,
    ),
]

MINIGAME_BY_ID: Final[dict[str, MinigameDefinition]] = {
    item.id: item for item in MINIGAME_REGISTRY
}
