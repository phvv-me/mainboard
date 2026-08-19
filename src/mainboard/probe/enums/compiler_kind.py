from enum import StrEnum, auto


class CompilerKind(StrEnum):
    """Compiler family a discovered binary belongs to."""

    CLANG = auto()
    CLANG_GRACE = "clang-grace"
    GCC = auto()
    NVCC = auto()
    UNKNOWN = auto()
