"""What `mb code app inspect` tells apart."""

from enum import StrEnum, auto
from pathlib import Path

# A result bundle's suffix, and the suffixes that name a built bundle even where none exists.
_RESULT = ".xcresult"
_BUILT = frozenset({".app", ".xcarchive"})


class Kind(StrEnum):
    """The kinds of path `mb code app inspect` reads, each with one inspector."""

    BUNDLE = auto()
    """a built bundle: an app, an archive, or any folder holding an Info.plist."""
    RESULTS = auto()
    """a result bundle, or a folder holding result bundles directly (Logs/Test, DerivedData)."""

    @classmethod
    def of(cls, path: Path) -> Kind | None:
        """The kind of `path`, `None` when it is of none."""
        if path.suffix in _BUILT:
            return cls.BUNDLE
        if path.suffix == _RESULT or any(path.glob(f"*{_RESULT}")):
            return cls.RESULTS
        if any(info.is_file() for info in (path / "Info.plist", path / "Contents" / "Info.plist")):
            return cls.BUNDLE
        return None
