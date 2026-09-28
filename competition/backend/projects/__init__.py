"""Competition project adapters live behind the contracts in this package."""

from .standard_2048 import Standard2048Adapter
from .tournament_variants import TOURNAMENT_ADAPTER_FACTORIES, TOURNAMENT_V2_ADAPTER_FACTORIES, tournament_project_catalog
from .cargo_transport import CargoTransportAdapter, CargoTransportAdapterV2
from .registry import ProjectRegistry

TOURNAMENT_ADAPTER_FACTORIES = (
    CargoTransportAdapter, CargoTransportAdapterV2,
    *TOURNAMENT_ADAPTER_FACTORIES, *TOURNAMENT_V2_ADAPTER_FACTORIES,
)

__all__ = [
    "ProjectRegistry",
    "Standard2048Adapter",
    "CargoTransportAdapter",
    "TOURNAMENT_ADAPTER_FACTORIES",
    "tournament_project_catalog",
]
