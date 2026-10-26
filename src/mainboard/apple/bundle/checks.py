"""What a built bundle must hold and must not, judged from what inspection read."""

import re
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Self

from cyclopts import Parameter
from patos import FrozenModel
from pydantic import ConfigDict, JsonValue, model_validator

from ...core.errors import MissionError

if TYPE_CHECKING:
    from collections.abc import Iterable


def _dotted(info: Mapping[str, JsonValue], key: str) -> JsonValue:
    """The value at dotted `key` inside `info`, None where the path leaves the dictionaries."""
    first, _, rest = key.partition(".")
    match info.get(first):
        case Mapping() as nested if rest:
            return _dotted(nested, rest)
        case found:
            return None if rest else found


def _pairs(values: Iterable[str]) -> dict[str, str]:
    """`KEY=VALUE` pairs as a mapping, refusing a repeated, empty or malformed key."""
    result: dict[str, str] = {}
    for value in values:
        key, separator, expected = value.partition("=")
        if not separator or key in result or not all(key.split(".")):
            raise MissionError("bundle expectations need unique KEY=VALUE pairs")
        result[key] = expected
    return result


@Parameter(name="*")
class BundleChecks(FrozenModel):
    """What a built bundle must hold and must not, every pattern checked before inspection."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    forbid_symbol: tuple[str, ...] = ()
    """repeatable case-insensitive regex over nm/otool stdout and stderr."""
    forbid_resource: tuple[str, ...] = ()
    """repeatable relative glob; ** also matches zero directories."""
    expect_info: tuple[str, ...] = ()
    """repeatable dotted Info.plist KEY=VALUE string expectations."""
    require_info: tuple[str, ...] = ()
    """repeatable dotted Info.plist keys requiring nonempty text."""
    expect_name: tuple[str, ...] = ()
    """repeatable LOCALE=VALUE display names in InfoPlist.strings."""

    @cached_property
    def expected(self) -> dict[str, str]:
        """The Info.plist values asked for, by dotted key."""
        return _pairs(self.expect_info)

    @cached_property
    def names(self) -> dict[str, str]:
        """The display names asked for, by locale."""
        return _pairs(self.expect_name)

    @cached_property
    def symbols(self) -> list[re.Pattern[str]]:
        """The forbidden-symbol patterns, compiled case-insensitive."""
        try:
            return [re.compile(pattern, re.IGNORECASE) for pattern in self.forbid_symbol]
        except re.PatternError as fault:
            raise MissionError(f"invalid forbidden-symbol pattern: {fault}") from fault

    @model_validator(mode="after")
    def _sound(self) -> Self:
        """Refuse a check that could never be read the way it was meant."""
        if not all(pattern.strip() for pattern in self.forbid_symbol + self.forbid_resource):
            raise MissionError("bundle exclusion patterns must not be empty")
        if any(
            PurePosixPath(glob).is_absolute() or ".." in PurePosixPath(glob).parts
            for glob in self.forbid_resource
        ):
            raise MissionError("resource globs must be relative to the bundle")
        if not all(re.fullmatch(r"[A-Za-z0-9_-]+", locale) for locale in self.names):
            raise MissionError("localized-name expectations need one locale directory name")
        if not all(key and all(key.split(".")) for key in self.require_info):
            raise MissionError("required Info.plist keys must be nonempty dotted paths")
        _ = self.expected, self.symbols
        return self

    def forbidden(self, bundle: Path, entries: Iterable[Path]) -> list[str]:
        """The entries under `bundle` a forbidden-resource glob matches."""
        return [
            f"forbidden resource: {entry.relative_to(bundle).as_posix()} ({pattern})"
            for entry in entries
            for pattern in self.forbid_resource
            if PurePosixPath(entry.relative_to(bundle).as_posix()).full_match(pattern)
        ]

    def misread(self, info: Mapping[str, JsonValue]) -> list[str]:
        """The Info.plist values that differ from those asked, and the required ones left empty."""
        differing = [
            f"Info.plist {key} does not equal {value!r}"
            for key, value in self.expected.items()
            if _dotted(info, key) != value
        ]
        empty = [
            f"Info.plist {key} is not nonempty text"
            for key in self.require_info
            if not (isinstance(found := _dotted(info, key), str) and found.strip())
        ]
        return differing + empty
