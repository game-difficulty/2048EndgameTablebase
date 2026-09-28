from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ..errors import CompetitionError
from .contracts import ProjectAdapter, ProjectDescriptor


AdapterFactory = Callable[[], ProjectAdapter]


class ProjectRegistry:
    """Exact-version registry used by the match orchestrator.

    Registration is process configuration. A competition freezes the returned
    descriptor snapshot, while a running session stores the exact registry key
    and version that created it.
    """

    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], AdapterFactory] = {}
        self._descriptors: dict[tuple[str, str], ProjectDescriptor] = {}

    def register(self, factory: AdapterFactory) -> None:
        adapter = factory()
        descriptor = adapter.descriptor
        key = (descriptor.project_ref, descriptor.rules_version)
        if key in self._factories:
            raise ValueError(f"duplicate project adapter: {key[0]}@{key[1]}")
        if adapter.project_id != descriptor.project_ref:
            raise ValueError("project adapter id does not match its descriptor")
        if adapter.rules_version != descriptor.rules_version:
            raise ValueError("project adapter version does not match its descriptor")
        self._factories[key] = factory
        self._descriptors[key] = descriptor

    def resolve(self, project_ref: str, rules_version: str) -> ProjectAdapter:
        key = (str(project_ref), str(rules_version))
        factory = self._factories.get(key)
        if factory is None:
            raise CompetitionError(
                "PROJECT_ADAPTER_UNAVAILABLE",
                f"Project adapter {key[0]}@{key[1]} is not registered.",
                409,
            )
        return factory()

    def descriptor(self, project_ref: str, rules_version: str) -> ProjectDescriptor:
        # Resolve first so callers get the public domain error rather than a KeyError.
        self.resolve(project_ref, rules_version)
        return self._descriptors[(str(project_ref), str(rules_version))]

    def snapshot(self, project_ref: str, rules_version: str) -> dict[str, Any]:
        return dict(self.descriptor(project_ref, rules_version).snapshot())

    def descriptors(self) -> Iterable[ProjectDescriptor]:
        return tuple(self._descriptors[key] for key in sorted(self._descriptors))
