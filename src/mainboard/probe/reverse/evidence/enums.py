from enum import IntEnum, StrEnum, auto


class ArtifactKind(StrEnum):
    native_elf = auto()
    native_macho = auto()
    fatbin = auto()
    cubin = auto()
    ptx = auto()
    # Breadth kinds (dotnet, asar, apk) belong to the later REA phase.


class Coverage(StrEnum):
    complete = auto()
    partial = auto()
    truncated = auto()
    failed = auto()
    not_run = auto()


class Confidence(StrEnum):
    observed = auto()
    inferred = auto()
    uncertain = auto()


class Provenance(IntEnum):
    """Merge competing provenance by max; ANALYSIS includes AI-derived findings."""

    DEFAULT = 0
    IMPORTED = auto()
    ANALYSIS = auto()
    USER_DEFINED = auto()
