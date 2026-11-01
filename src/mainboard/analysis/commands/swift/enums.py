from enum import StrEnum, auto


class SourceKind(StrEnum):
    production = auto()
    test = auto()
    tooling = auto()
    generated = auto()
    vendored = auto()


class SyntaxStatus(StrEnum):
    pending = auto()
    parsed = auto()
    parse_error = auto()
    excluded = auto()
    not_swift = auto()
