"""Inspect built Apple bundles without executing their code or changing their contents."""

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import JsonValue, TypeAdapter, ValidationError

from ...core.errors import MissionError
from ...core.execution.tool import Runner, ToolOutput
from .checks import BundleChecks

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ...render.values import Node

# The most a bundle may hold before inspection refuses it: entries, Mach-O files, fat slices.
_ENTRIES = 10000
_BINARIES = 128
_SLICES = 16
# The magic numbers opening a thin Mach-O file, and a fat one in either byte order.
_THIN = frozenset(
    bytes.fromhex(value) for value in ("feedface", "feedfacf", "cefaedfe", "cffaedfe")
)
_FAT = {"cafebabe": "big", "cafebabf": "big", "bebafeca": "little", "bfbafeca": "little"}
# The directory a macOS bundle keeps everything in; an iOS one keeps it at its root.
_CONTENTS = "Contents"
# The lowest Java class-file version, which shares the fat magic.
_JAVA = 45


def _digest(path: Path) -> str:
    """The SHA-256 of `path`'s bytes."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class BinaryInspection(FrozenModel):
    """One Mach-O file of a bundle: its digest, undefined symbols and linked libraries."""

    path: str
    sha256: str
    symbols: ToolOutput
    libraries: ToolOutput

    def matched(self, patterns: Iterable[re.Pattern[str]]) -> list[str]:
        """Each pattern either tool's output matched, said once per tool."""
        said = {"nm": self.symbols, "otool": self.libraries}
        return [
            f"{self.path}: {tool} matched {found.group()!r}"
            for tool, output in said.items()
            for pattern in patterns
            if (found := pattern.search(output.stdout + "\n" + output.stderr))
        ]


class BundleReport(FrozenModel):
    """What a bundle holds and every check it failed."""

    bundle: Path
    info_sha256: str
    info: dict[str, JsonValue]
    localized_names: dict[str, str]
    localized_sha256: dict[str, str]
    file_count: int
    binaries: list[BinaryInspection]
    violations: list[str]

    @property
    def successful(self) -> bool:
        return bool(self.binaries) and not self.violations

    def document(self) -> dict[str, Node]:
        return {"successful": self.successful, **self.model_dump(mode="json")}


class _Localized(FrozenModel):
    """The display names a bundle's locales declare, their files' digests, and the mismatches."""

    names: dict[str, str]
    digests: dict[str, str]
    violations: list[str]


# A bundle checked against nothing but its own shape.
_UNCHECKED = BundleChecks()


class AppleBundle:
    """One built bundle read through the selected Xcode's tools, its code never run."""

    @staticmethod
    def _entries(bundle: Path) -> list[Path]:
        """Every entry under `bundle`, refusing too many or a link that escapes it."""
        entries: list[Path] = []
        for entry in bundle.rglob("*"):
            if len(entries) >= _ENTRIES:
                raise MissionError("bundle inspection exceeds 10,000 entries")
            if entry.is_symlink() and not entry.resolve(strict=True).is_relative_to(bundle):
                raise MissionError(f"bundle symlink escapes its root: {entry}")
            entries.append(entry)
        return sorted(entries)

    @staticmethod
    def _macho(path: Path) -> bool:
        """Whether `path` is a thin or fat Mach-O file, refusing a fat header past 16 slices."""
        with path.open("rb") as source:
            header = source.read(8)
        if header[:4] in _THIN:
            return True
        if (order := _FAT.get(header[:4].hex())) is None:
            return False
        count = int.from_bytes(header[4:], "big" if order == "big" else "little")
        # Java class files share the fat magic, but their version is not an architecture count.
        if path.suffix == ".class" and len(header) == 8 and count >= _JAVA:
            return False
        if len(header) != 8 or not 0 < count <= _SLICES:
            raise MissionError(f"invalid or unsupported fat Mach-O header at {path}")
        return True

    @staticmethod
    def _unexecutable(
        info: Mapping[str, JsonValue], bundle: Path, binaries: Iterable[BinaryInspection]
    ) -> list[str]:
        """The violation of an executable that names no inspected Mach-O file, if any."""
        executable = info.get("CFBundleExecutable")
        contents = bundle / _CONTENTS if (bundle / _CONTENTS).is_dir() else bundle
        home = contents / "MacOS" if contents != bundle else bundle
        found = (home / executable).resolve() if isinstance(executable, str) else None
        if found is not None and found in {bundle / item.path for item in binaries}:
            return []
        return ["CFBundleExecutable does not identify an inspected Mach-O file"]

    def __init__(self, runner: Runner) -> None:
        self.runner = runner

    def _binaries(
        self, bundle: Path, files: Iterable[Path], checks: BundleChecks
    ) -> tuple[list[BinaryInspection], list[str]]:
        """Every Mach-O file inspected, and each forbidden symbol its tools printed."""
        binaries: list[BinaryInspection] = []
        for file in files:
            if not AppleBundle._macho(file):
                continue
            if len(binaries) >= _BINARIES:
                raise MissionError("bundle inspection exceeds 128 Mach-O files")
            binaries.append(self._inspected(file, bundle=bundle))
        return binaries, [found for binary in binaries for found in binary.matched(checks.symbols)]

    def _inspected(self, file: Path, *, bundle: Path) -> BinaryInspection:
        """One Mach-O file's undefined symbols and linked libraries, every architecture."""
        undefined = self._native("nm", "-arch", "all", "-u", str(file))
        libraries = self._native("otool", "-arch", "all", "-L", str(file))
        if not libraries.stdout.strip():
            raise MissionError(f"otool returned no inspection for {file}")
        return BinaryInspection(
            path=file.relative_to(bundle).as_posix(),
            sha256=_digest(file),
            symbols=undefined,
            libraries=libraries,
        )

    def _localized(self, resources: Path, checks: BundleChecks) -> _Localized:
        """Each asked locale's declared display name, its file's digest, and any mismatch."""
        names: dict[str, str] = {}
        digests: dict[str, str] = {}
        violations: list[str] = []
        for locale, expected in checks.names.items():
            source = resources / f"{locale}.lproj/InfoPlist.strings"
            name = self._plist(source).get("CFBundleDisplayName")
            digests[locale] = _digest(source)
            if isinstance(name, str):
                names[locale] = name
            if name != expected:
                violations.append(f"{locale} display name does not equal {expected!r}")
        return _Localized(names=names, digests=digests, violations=violations)

    def _native(self, *arguments: str) -> ToolOutput:
        """One xcrun tool's complete output, refusing a failure or a truncation."""
        result = self.runner.run("/usr/bin/xcrun", arguments)
        if result.returncode or result.truncated:
            raise MissionError(
                f"bundle inspection {arguments[0]} failed "
                f"(exit {result.returncode}, truncated={result.truncated}): "
                f"{result.stderr.strip()}"
            )
        return result

    def _plist(self, path: Path) -> dict[str, JsonValue]:
        """The property list at `path` as JSON, through plutil."""
        result = self._native("plutil", "-convert", "json", "-o", "-", str(path))
        try:
            return TypeAdapter(dict[str, JsonValue]).validate_json(result.stdout, strict=True)
        except ValidationError as fault:
            raise MissionError(f"invalid bundle property list at {path}: {fault}") from fault

    def inspect(self, path: Path, checks: BundleChecks = _UNCHECKED) -> BundleReport:
        """Read the bundle at `path` and judge it against `checks`."""
        try:
            bundle = path.expanduser().resolve(strict=True)
        except FileNotFoundError as fault:
            raise MissionError(f"no Apple bundle directory at {path}") from fault
        if not bundle.is_dir():
            raise MissionError(f"no Apple bundle directory at {bundle}")
        entries = AppleBundle._entries(bundle)
        files = sorted({entry.resolve(strict=True) for entry in entries if entry.is_file()})
        contents = bundle / _CONTENTS if (bundle / _CONTENTS).is_dir() else bundle
        info = self._plist(contents / "Info.plist")
        localized = self._localized(
            contents / "Resources" if contents != bundle else bundle, checks
        )
        binaries, matched = self._binaries(bundle, files, checks)
        violations = [
            *checks.forbidden(bundle, entries),
            *checks.misread(info),
            *localized.violations,
            *matched,
            *AppleBundle._unexecutable(info, bundle, binaries),
        ]
        return BundleReport(
            bundle=bundle,
            info_sha256=_digest(contents / "Info.plist"),
            info=info,
            localized_names=localized.names,
            localized_sha256=localized.digests,
            file_count=len(files),
            binaries=binaries,
            violations=violations,
        )
