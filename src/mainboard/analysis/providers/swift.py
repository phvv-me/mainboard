"""Current-toolchain SwiftSyntax facts, compiled offline and cached by code/toolchain identity."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self

from patos import FrozenModel
from pydantic import TypeAdapter, ValidationError

from ...core.errors import MissionError
from ...core.execution.tool import Runner, ToolRunner
from ..engine.facts import listed
from .constants import SYNTAX_TABLES, TABLES
from .external import Row, Table, Tool

# The helper's source, compiled once per toolchain.
_SOURCE = Path(__file__).with_name("swift_syntax.swift")


class SwiftToolchain(FrozenModel):
    """The selected parser build inputs, not the analyzed app's target SDK or type-check result."""

    compiler: str
    compiler_version: str
    sdk: str | None
    host_libraries: str
    helper_identity: str
    helper_source_sha256: str


class _Build(FrozenModel):
    """The SwiftSyntax helper as this toolchain builds it, cached by everything that shapes it.

    version: what `swiftc --version` prints, as printed.
    sdk: the `-sdk` flags of a macOS build, none elsewhere.
    host: the toolchain's SwiftSyntax host libraries.
    """

    compiler: Path
    version: str
    sdk: tuple[str, ...]
    host: Path

    @classmethod
    def found(cls, runner: Runner) -> Self:
        """The selected toolchain, refused when it cannot build the helper."""
        compiler = _Build._compiler(runner)
        host = compiler.resolve().parent.parent / "lib/swift/host"
        if not (host / "SwiftParser.swiftmodule").is_dir():
            raise MissionError(f"Swift analysis requires SwiftParser host libraries at {host}")
        sdk = _Build._sdk(runner) if sys.platform == "darwin" else ()
        version = runner.run(str(compiler), ("--version",))
        if version.returncode or version.truncated or not version.stdout.strip():
            raise MissionError(f"Cannot identify Swift compiler: {version.stderr}")
        return cls(compiler=compiler, version=version.stdout, sdk=sdk, host=host)

    @property
    def digest(self) -> str:
        """The helper's identity: its source, the compiler, its SDK and its host libraries."""
        identity = (
            _SOURCE.read_bytes()
            + str(self.compiler.resolve()).encode()
            + self.version.encode()
            + str(self.sdk).encode()
        )
        for file in sorted(
            self.host.glob("*Swift*.dylib" if sys.platform == "darwin" else "*Swift*.so")
        ):
            identity += f"{file.name}:{file.stat().st_size}:{file.stat().st_mtime_ns}".encode()
        return hashlib.sha256(identity).hexdigest()[:24]

    @property
    def folder(self) -> Path:
        return Path.home() / ".cache/mainboard/analysis/swift" / self.digest

    @staticmethod
    def _compiler(runner: Runner) -> Path:
        """`swiftc`: Xcode's selected one on macOS, the one on PATH elsewhere."""
        if sys.platform != "darwin":
            if found := shutil.which("swiftc"):
                return Path(found)
            raise MissionError("Swift analysis requires swiftc with SwiftSyntax host libraries")
        found_xcode = runner.run("/usr/bin/xcrun", ("--find", "swiftc"))
        compiler = Path(found_xcode.stdout.strip())
        if found_xcode.returncode or found_xcode.truncated or not compiler.is_file():
            raise MissionError("Swift analysis requires the selected Xcode Swift toolchain")
        return compiler

    @staticmethod
    def _sdk(runner: Runner) -> tuple[str, ...]:
        """The `-sdk` flags naming the selected macOS SDK."""
        selected = runner.run("/usr/bin/xcrun", ("--sdk", "macosx", "--show-sdk-path"))
        path = selected.stdout.strip()
        if selected.returncode or selected.truncated or not path or not Path(path).is_dir():
            raise MissionError(f"Cannot resolve macOS SDK: {selected.stderr}")
        return ("-sdk", str(Path(path).resolve()))

    def compiled(self, runner: Runner, helper: Path) -> None:
        """Compile the helper into `helper`, published only once it built."""
        with tempfile.TemporaryDirectory(dir=helper.parent) as scratch:
            output = Path(scratch) / "swift-syntax"
            source = Path(scratch) / _SOURCE.name
            source.write_bytes(_SOURCE.read_bytes())
            host = str(self.host)
            linked = ("-I", host, "-L", host, "-Xlinker", "-rpath", "-Xlinker", host)
            # The compiler's own module cache lives outside the sandboxes this build runs in (a
            # fresh CI runner has none to read and may write none), so the build keeps its own.
            cached = ("-module-cache-path", str(Path(scratch) / "modules"))
            built = runner.run(
                str(self.compiler), (*linked, *cached, *self.sdk, str(source), "-o", str(output))
            )
            if built.returncode or built.truncated:
                raise MissionError(f"SwiftSyntax helper compilation failed: {built.stderr}")
            os.replace(output, helper)

    def recorded(self, folder: Path) -> None:
        """Write what the helper in `folder` was built with, replaced in one step."""
        provenance = SwiftToolchain(
            compiler=str(self.compiler.resolve()),
            compiler_version=self.version.strip(),
            sdk=self.sdk[1] if self.sdk else None,
            host_libraries=str(self.host),
            helper_identity=folder.name,
            helper_source_sha256=hashlib.sha256(_SOURCE.read_bytes()).hexdigest(),
        )
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=folder, suffix=".json", delete=False
        ) as record:
            record.write(provenance.model_dump_json())
        os.replace(record.name, folder / "toolchain.json")


class SwiftSource(Tool):
    """Syntax evidence only: imports and named APIs are not compiler-resolved symbols."""

    family: ClassVar[str] = "SwiftSource"

    @staticmethod
    def _covered(facts: Sequence[Row], files: list[str]) -> dict[str, int]:
        """Each requested file's gap count, refused unless every file is covered exactly once."""
        paths = [row[1] for row in facts]
        if len(paths) != len(files) or set(paths) != set(files):
            raise MissionError("SwiftSyntax must cover every requested file exactly once")
        gaps: dict[str, int] = {}
        for fact_id, path, language, line, column, count, message in facts:
            if (
                not isinstance(path, str)
                or fact_id != f"swift:{path}"
                or language != "swift"
                or line != 1
                or column != 0
                or not isinstance(count, int)
                or count < 0
                or not isinstance(message, str)
                or (count == 0) != (message == "")
            ):
                raise MissionError("SwiftSyntax emitted invalid file coverage metadata")
            gaps[path] = count
        return gaps

    @staticmethod
    def _helper(runner: Runner) -> Path:
        """The helper for the selected toolchain, built once per identity."""
        build = _Build.found(runner)
        folder = build.folder
        folder.mkdir(parents=True, exist_ok=True)
        build.recorded(folder)
        helper = folder / "swift-syntax"
        if not helper.is_file():
            build.compiled(runner, helper)
        return helper

    @staticmethod
    def _placed(row: Row, gaps: Mapping[str, int]) -> bool:
        """Whether a syntax row sits at a real place in a file parsed without gaps."""
        fact_id, path, language, line, column, *_ = row
        return (
            isinstance(path, str)
            and path in gaps
            and not gaps[path]
            and fact_id == f"swift:{path}"
            and language == "swift"
            and isinstance(line, int)
            and line >= 1
            and isinstance(column, int)
            and column >= 0
        )

    @staticmethod
    def _typed(rows: dict[str, list[Row]]) -> None:
        """Refuse a table that is missing or unexpected, or a row of the wrong shape."""
        if set(rows) != set(SYNTAX_TABLES):
            raise MissionError("SwiftSyntax tables are missing or unexpected; coverage is unknown")
        for table, columns in SYNTAX_TABLES.items():
            if any(
                len(row) != len(columns)
                or any(
                    not isinstance(value, str if kind == "VARCHAR" else int)
                    for value, kind in zip(row, columns.values(), strict=False)
                )
                for row in rows[table]
            ):
                raise MissionError(f"SwiftSyntax emitted an invalid {table} row")

    @staticmethod
    def _validate(rows: dict[str, list[Row]], files: list[str]) -> None:
        SwiftSource._typed(rows)
        gaps = SwiftSource._covered(rows["facts"], files)
        for table in SYNTAX_TABLES.keys() - {"facts"}:
            if not all(SwiftSource._placed(row, gaps) for row in rows[table]):
                raise MissionError(f"SwiftSyntax emitted uncovered/partial syntax in {table}")

    def _analyzed(
        self, runner: Runner, root: Path, files: list[str]
    ) -> tuple[dict[str, list[Row]], Path]:
        """The helper's tables for `files` under `root`, validated, and the helper that ran."""
        helper = self._helper(runner)
        with tempfile.TemporaryDirectory(prefix="mb-swift-") as scratch:
            request = Path(scratch) / "request.json"
            request.write_text(json.dumps({"root": str(root), "files": files}), encoding="utf-8")
            result = runner.run(str(helper), (str(request),), cwd=root)
        if result.returncode or result.truncated:
            raise MissionError(f"SwiftSyntax analysis failed: {result.stderr or result.stdout}")
        try:
            rows = TypeAdapter(dict[str, list[Row]]).validate_json(result.stdout, strict=True)
        except ValidationError as error:
            raise MissionError("SwiftSyntax emitted invalid result data") from error
        self._validate(rows, files)
        return rows, helper

    def run(self, root: Path) -> dict[str, Table]:
        root = root.resolve()
        files = [path for path in listed(root) if Path(path).suffix.lower() == ".swift"]
        if not files:
            return {name: (columns, []) for name, columns in TABLES.items()}
        rows, helper = self._analyzed(ToolRunner(timeout=120), root, files)
        try:
            provenance = SwiftToolchain.model_validate_json(
                (helper.parent / "toolchain.json").read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise MissionError("SwiftSyntax toolchain provenance is missing or invalid") from error
        rows["toolchain"] = [
            (
                provenance.compiler,
                provenance.compiler_version,
                provenance.sdk,
                provenance.host_libraries,
                provenance.helper_identity,
                provenance.helper_source_sha256,
            )
        ]
        return {name: (columns, rows[name]) for name, columns in TABLES.items()}
