"""Current-toolchain SwiftSyntax facts, compiled offline and cached by code/toolchain identity."""

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

from pydantic import TypeAdapter

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
TABLES = {
    "facts": PLACE | {"parser_gaps": "BIGINT", "parser_message": "VARCHAR"},
    "imports": PLACE | {"name": "VARCHAR"},
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
    "calls": PLACE | {"name": "VARCHAR", "receiver": "VARCHAR", "shadowed": "BIGINT"},
    "references": PLACE | {"name": "VARCHAR", "receiver": "VARCHAR", "shadowed": "BIGINT"},
    "states": PLACE | {"name": "VARCHAR", "attribute": "VARCHAR", "is_private": "BIGINT"},
    "erasures": PLACE | {"view_name": "VARCHAR"},
}


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
        rows = TypeAdapter(dict[str, list[Row]]).validate_json(result.stdout)
        return {name: (columns, rows.get(name, [])) for name, columns in TABLES.items()}

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
            if selected_sdk.returncode or selected_sdk.truncated:
                raise MissionError(f"Cannot resolve macOS SDK: {selected_sdk.stderr}")
            sdk = ("-sdk", selected_sdk.stdout.strip())
        version = runner.run(str(compiler), ("--version",))
        if version.returncode or version.truncated:
            raise MissionError(f"Cannot identify Swift compiler: {version.stderr}")
        source = Path(__file__).with_name("swift_syntax.swift")
        identity = (
            source.read_bytes()
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
        if helper.is_file():
            return helper
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as scratch:
            output = Path(scratch) / "swift-syntax"
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
                    str(source),
                    "-o",
                    str(output),
                ),
            )
            if built.returncode or built.truncated:
                raise MissionError(f"SwiftSyntax helper compilation failed: {built.stderr}")
            os.replace(output, helper)
        return helper
