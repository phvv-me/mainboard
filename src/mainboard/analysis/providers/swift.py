"""Current-toolchain SwiftSyntax facts, compiled offline and cached by code/toolchain identity."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

from patos import FrozenModel
from pydantic import TypeAdapter, ValidationError

from ...core.errors import MissionError
from ...core.tool import Runner, ToolRunner
from ..facts import listed
from .external import Row, Table, Tool

PLACE = {
    "fact_id": "VARCHAR",
    "path": "VARCHAR",
    "language": "VARCHAR",
    "start_line": "BIGINT",
    "start_column": "BIGINT",
}
SYNTAX_TABLES = {
    "facts": PLACE | {"parser_gaps": "BIGINT", "parser_message": "VARCHAR"},
    "imports": PLACE | {"name": "VARCHAR"},
    "conditions": PLACE | {"condition": "VARCHAR"},
    "conformances": PLACE
    | {
        "owner_id": "BIGINT",
        "owner": "VARCHAR",
        "owner_kind": "VARCHAR",
        "inherited": "VARCHAR",
        "shadowed": "BIGINT",
    },
    "attributes": PLACE
    | {
        "owner_id": "BIGINT",
        "owner": "VARCHAR",
        "attribute": "VARCHAR",
        "container_id": "BIGINT",
        "container": "VARCHAR",
        "container_kind": "VARCHAR",
        "is_static": "BIGINT",
        "shadowed": "BIGINT",
    },
    "attribute_options": PLACE
    | {
        "owner_id": "BIGINT",
        "owner": "VARCHAR",
        "container_id": "BIGINT",
        "container": "VARCHAR",
        "container_kind": "VARCHAR",
        "is_static": "BIGINT",
        "attribute": "VARCHAR",
        "attribute_shadowed": "BIGINT",
        "option": "VARCHAR",
        "receiver": "VARCHAR",
        "shadowed": "BIGINT",
    },
    "calls": PLACE | {"name": "VARCHAR", "receiver": "VARCHAR", "shadowed": "BIGINT"},
    "references": PLACE | {"name": "VARCHAR", "receiver": "VARCHAR", "shadowed": "BIGINT"},
    "states": PLACE | {"name": "VARCHAR", "attribute": "VARCHAR", "is_private": "BIGINT"},
    "erasures": PLACE | {"view_name": "VARCHAR"},
}
TABLES = SYNTAX_TABLES | {
    "toolchain": {
        "compiler": "VARCHAR",
        "compiler_version": "VARCHAR",
        "sdk": "VARCHAR",
        "host_libraries": "VARCHAR",
        "helper_identity": "VARCHAR",
        "helper_source_sha256": "VARCHAR",
    },
}


class SwiftToolchain(FrozenModel):
    """The selected parser build inputs, not the analyzed app's target SDK or type-check result."""

    compiler: str
    compiler_version: str
    sdk: str | None
    host_libraries: str
    helper_identity: str
    helper_source_sha256: str


class SwiftSource(Tool):
    """Syntax evidence only: imports and named APIs are not compiler-resolved symbols."""

    family: ClassVar[str] = "SwiftSource"

    def run(self, root: Path) -> dict[str, Table]:
        root = root.resolve()
        files = [path for path in listed(root) if Path(path).suffix.lower() == ".swift"]
        if not files:
            return {name: (columns, []) for name, columns in TABLES.items()}
        runner = ToolRunner(timeout=120)
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

    @staticmethod
    def _validate(rows: dict[str, list[Row]], files: list[str]) -> None:
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
        paths = [row[1] for row in rows["facts"]]
        if len(paths) != len(files) or set(paths) != set(files):
            raise MissionError("SwiftSyntax must cover every requested file exactly once")
        gaps: dict[str, int] = {}
        for fact_id, path, language, line, column, count, message in rows["facts"]:
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
        for table in SYNTAX_TABLES.keys() - {"facts"}:
            for fact_id, path, language, line, column, *_ in rows[table]:
                if (
                    not isinstance(path, str)
                    or path not in gaps
                    or gaps[path]
                    or fact_id != f"swift:{path}"
                    or language != "swift"
                    or not isinstance(line, int)
                    or line < 1
                    or not isinstance(column, int)
                    or column < 0
                ):
                    raise MissionError(f"SwiftSyntax emitted uncovered/partial syntax in {table}")

    @staticmethod
    def _helper(runner: Runner) -> Path:
        if sys.platform == "darwin":
            found = runner.run("/usr/bin/xcrun", ("--find", "swiftc"))
            compiler = Path(found.stdout.strip())
            if found.returncode or found.truncated or not compiler.is_file():
                raise MissionError("Swift analysis requires the selected Xcode Swift toolchain")
        else:
            found_compiler = shutil.which("swiftc")
            if not found_compiler:
                raise MissionError(
                    "Swift analysis requires swiftc with SwiftSyntax host libraries"
                )
            compiler = Path(found_compiler)
        host = compiler.resolve().parent.parent / "lib/swift/host"
        if not (host / "SwiftParser.swiftmodule").is_dir():
            raise MissionError(f"Swift analysis requires SwiftParser host libraries at {host}")
        sdk: tuple[str, ...] = ()
        if sys.platform == "darwin":
            selected_sdk = runner.run("/usr/bin/xcrun", ("--sdk", "macosx", "--show-sdk-path"))
            if (
                selected_sdk.returncode
                or selected_sdk.truncated
                or not selected_sdk.stdout.strip()
                or not Path(selected_sdk.stdout.strip()).is_dir()
            ):
                raise MissionError(f"Cannot resolve macOS SDK: {selected_sdk.stderr}")
            sdk = ("-sdk", str(Path(selected_sdk.stdout.strip()).resolve()))
        version = runner.run(str(compiler), ("--version",))
        if version.returncode or version.truncated or not version.stdout.strip():
            raise MissionError(f"Cannot identify Swift compiler: {version.stderr}")
        source = Path(__file__).with_name("swift_syntax.swift")
        source_bytes = source.read_bytes()
        identity = (
            source_bytes
            + str(compiler.resolve()).encode()
            + version.stdout.encode()
            + str(sdk).encode()
        )
        for file in sorted(
            host.glob("*Swift*.dylib" if sys.platform == "darwin" else "*Swift*.so")
        ):
            identity += f"{file.name}:{file.stat().st_size}:{file.stat().st_mtime_ns}".encode()
        digest = hashlib.sha256(identity).hexdigest()[:24]
        folder = Path.home() / ".cache/mainboard/analysis/swift" / digest
        helper = folder / "swift-syntax"
        folder.mkdir(parents=True, exist_ok=True)
        provenance = SwiftToolchain(
            compiler=str(compiler.resolve()),
            compiler_version=version.stdout.strip(),
            sdk=sdk[1] if sdk else None,
            host_libraries=str(host),
            helper_identity=digest,
            helper_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        )
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=folder, suffix=".json", delete=False
        ) as record:
            record.write(provenance.model_dump_json())
        os.replace(record.name, folder / "toolchain.json")
        if helper.is_file():
            return helper
        with tempfile.TemporaryDirectory(dir=folder) as scratch:
            output = Path(scratch) / "swift-syntax"
            source_copy = Path(scratch) / source.name
            source_copy.write_bytes(source_bytes)
            built = runner.run(
                str(compiler),
                (
                    "-I",
                    str(host),
                    "-L",
                    str(host),
                    "-Xlinker",
                    "-rpath",
                    "-Xlinker",
                    str(host),
                    *sdk,
                    str(source_copy),
                    "-o",
                    str(output),
                ),
            )
            if built.returncode or built.truncated:
                raise MissionError(f"SwiftSyntax helper compilation failed: {built.stderr}")
            os.replace(output, helper)
        return helper
