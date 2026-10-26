import abc
from collections.abc import Mapping
from typing import ClassVar

from patos import FrozenModel, Registry

from ...core.errors import MissionError
from .models import Artifact, ArtifactKind, DeviceResult, Evidence, JsonValue


class Availability(FrozenModel):
    ok: bool
    reason: str = ""
    remediation: str = ""


class Effects(FrozenModel):
    read_only: bool
    executes_foreign_code: bool
    destructive: bool


class Request(FrozenModel):
    artifact: Artifact
    operation: str
    params: Mapping[str, JsonValue] = {}
    arch: str = ""
    function: str = ""

    def options(self) -> dict[str, JsonValue]:
        """Selectors participate in the cache identity even when applied after parsing."""
        options = dict(self.params)
        for key, value in (("arch", self.arch), ("function", self.function)):
            if key in options and options[key] != value:
                raise ValueError(f"conflicting {key} selector in params")
            if value:
                options[key] = value
        return options


class ReverseProvider(Registry, abc.ABC):
    tool: ClassVar[str]
    kinds: ClassVar[frozenset[str]]
    artifacts: ClassVar[frozenset[ArtifactKind]]
    operations: ClassVar[Mapping[str, Effects]]

    @abc.abstractmethod
    def analyze(self, request: Request) -> Evidence[DeviceResult]: ...

    @abc.abstractmethod
    def available(self) -> Availability: ...

    @classmethod
    def supporting(cls, kind: str, artifact: ArtifactKind, operation: str) -> ReverseProvider:
        reasons: list[str] = []
        for provider_type in cls.implementations():
            if (
                kind not in provider_type.kinds
                or artifact not in provider_type.artifacts
                or operation not in provider_type.operations
            ):
                continue
            provider = provider_type()
            availability = provider.available()
            if availability.ok:
                return provider
            reasons.append(f"{provider.tool}: {availability.reason}; {availability.remediation}")
        detail = "; ".join(reasons) or "no provider supports that artifact and operation"
        raise MissionError(f"cannot analyze {artifact}/{operation}: {detail}")
