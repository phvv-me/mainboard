"""Inspect built Apple bundles without executing their code or changing their contents."""

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from patos import FrozenModel
from pydantic import JsonValue, TypeAdapter, ValidationError

from .core.errors import MissionError
from .core.tool import Runner, ToolOutput

if TYPE_CHECKING:
    from .render.values import Node


class BinaryInspection(FrozenModel):
    path: str
    sha256: str
    symbols: ToolOutput
    libraries: ToolOutput


class BundleReport(FrozenModel):
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


class AppleBundle:
    def __init__(self, runner: Runner) -> None:
        self.runner = runner

    def inspect(
        self,
        path: Path,
        *,
        forbid_symbol: tuple[str, ...] = (),
        forbid_resource: tuple[str, ...] = (),
        expect_info: tuple[str, ...] = (),
        require_info: tuple[str, ...] = (),
        expect_name: tuple[str, ...] = (),
    ) -> BundleReport:
        expected = self._pairs(expect_info)
        names = self._pairs(expect_name)
        try:
            symbols = [re.compile(pattern, re.IGNORECASE) for pattern in forbid_symbol]
        except re.PatternError as fault:
            raise MissionError(f"invalid forbidden-symbol pattern: {fault}") from fault
        if any(not pattern.strip() for pattern in forbid_symbol + forbid_resource):
            raise MissionError("bundle exclusion patterns must not be empty")
        if any(
            PurePosixPath(pattern).is_absolute() or ".." in PurePosixPath(pattern).parts
            for pattern in forbid_resource
        ):
            raise MissionError("resource globs must be relative to the bundle")
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", locale) for locale in names):
            raise MissionError("localized-name expectations need one locale directory name")
        if any(not key or any(not part for part in key.split(".")) for key in require_info):
            raise MissionError("required Info.plist keys must be nonempty dotted paths")
        bundle = path.expanduser().resolve(strict=True)
        if not bundle.is_dir():
            raise MissionError(f"no Apple bundle directory at {bundle}")
        entries: list[Path] = []
        for entry in bundle.rglob("*"):
            if len(entries) >= 10000:
                raise MissionError("bundle inspection exceeds 10,000 entries")
            if entry.is_symlink() and not entry.resolve(strict=True).is_relative_to(bundle):
                raise MissionError(f"bundle symlink escapes its root: {entry}")
            entries.append(entry)
        entries.sort()
        files = sorted({entry.resolve(strict=True) for entry in entries if entry.is_file()})
        contents = bundle / "Contents" if (bundle / "Contents").is_dir() else bundle
        info_path = contents / "Info.plist"
        info = self._plist(info_path)
        violations = [
            f"forbidden resource: {entry.relative_to(bundle).as_posix()} ({pattern})"
            for entry in entries
            for pattern in forbid_resource
            if PurePosixPath(entry.relative_to(bundle).as_posix()).full_match(pattern)
        ]
        for key, value in expected.items():
            if self._value(info, key) != value:
                violations.append(f"Info.plist {key} does not equal {value!r}")
        for key in require_info:
            found = self._value(info, key)
            if not isinstance(found, str) or not found.strip():
                violations.append(f"Info.plist {key} is not nonempty text")
        resources = contents / "Resources" if contents != bundle else bundle
        localized: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for locale, expected_name in names.items():
            source = resources / f"{locale}.lproj/InfoPlist.strings"
            localized_info = self._plist(source)
            hashes[locale] = self._digest(source)
            name = localized_info.get("CFBundleDisplayName")
            if isinstance(name, str):
                localized[locale] = name
            if name != expected_name:
                violations.append(f"{locale} display name does not equal {expected_name!r}")
        binaries: list[BinaryInspection] = []
        for file in files:
            if not self._macho(file):
                continue
            if len(binaries) >= 128:
                raise MissionError("bundle inspection exceeds 128 Mach-O files")
            undefined = self._native("nm", "-arch", "all", "-u", str(file))
            libraries = self._native("otool", "-arch", "all", "-L", str(file))
            if not libraries.stdout.strip():
                raise MissionError(f"otool returned no inspection for {file}")
            relative = file.relative_to(bundle).as_posix()
            for tool, result in [("nm", undefined), ("otool", libraries)]:
                for pattern in symbols:
                    if match := pattern.search(result.stdout + "\n" + result.stderr):
                        violations.append(f"{relative}: {tool} matched {match.group()!r}")
            binaries.append(
                BinaryInspection(
                    path=relative,
                    sha256=self._digest(file),
                    symbols=undefined,
                    libraries=libraries,
                )
            )
        executable = info.get("CFBundleExecutable")
        executable_path = (
            (contents / "MacOS" / executable if contents != bundle else bundle / executable)
            if isinstance(executable, str)
            else None
        )
        if (
            executable_path is None
            or not executable_path.resolve().is_relative_to(bundle)
            or executable_path.resolve() not in {bundle / item.path for item in binaries}
        ):
            violations.append("CFBundleExecutable does not identify an inspected Mach-O file")
        return BundleReport(
            bundle=bundle,
            info_sha256=self._digest(info_path),
            info=info,
            localized_names=localized,
            localized_sha256=hashes,
            file_count=len(files),
            binaries=binaries,
            violations=violations,
        )

    def _plist(self, path: Path) -> dict[str, JsonValue]:
        result = self._native("plutil", "-convert", "json", "-o", "-", str(path))
        try:
            return TypeAdapter(dict[str, JsonValue]).validate_json(result.stdout, strict=True)
        except ValidationError as fault:
            raise MissionError(f"invalid bundle property list at {path}: {fault}") from fault

    def _native(self, *arguments: str) -> ToolOutput:
        result = self.runner.run("/usr/bin/xcrun", arguments)
        if result.returncode or result.truncated:
            raise MissionError(
                f"bundle inspection {arguments[0]} failed "
                f"(exit {result.returncode}, truncated={result.truncated}): "
                f"{result.stderr.strip()}"
            )
        return result

    @staticmethod
    def _pairs(values: tuple[str, ...]) -> dict[str, str]:
        result: dict[str, str] = {}
        for value in values:
            key, separator, expected = value.partition("=")
            if (
                not separator
                or not key
                or key in result
                or any(not part for part in key.split("."))
            ):
                raise MissionError("bundle expectations need unique KEY=VALUE pairs")
            result[key] = expected
        return result

    @staticmethod
    def _value(info: dict[str, JsonValue], key: str) -> JsonValue:
        value: JsonValue = info
        for part in key.split("."):
            if not isinstance(value, dict):
                return None
            value = value.get(part)
        return value

    @staticmethod
    def _digest(path: Path) -> str:
        with path.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()

    @staticmethod
    def _macho(path: Path) -> bool:
        with path.open("rb") as source:
            header = source.read(8)
        if header[:4] in {
            bytes.fromhex(value) for value in ["feedface", "feedfacf", "cefaedfe", "cffaedfe"]
        }:
            return True
        if header[:4].hex() in {"cafebabe", "cafebabf"}:
            count = int.from_bytes(header[4:], "big")
        elif header[:4].hex() in {"bebafeca", "bfbafeca"}:
            count = int.from_bytes(header[4:], "little")
        else:
            return False
        # Java class files share the fat magic, but their version is not an architecture count.
        if path.suffix == ".class" and len(header) == 8 and count >= 45:
            return False
        if len(header) != 8 or not 0 < count <= 16:
            raise MissionError(f"invalid or unsupported fat Mach-O header at {path}")
        return True
