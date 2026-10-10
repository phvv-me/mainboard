import contextlib
import importlib.metadata
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from cyclopts import App
from guard import Guard

import mainboard


class Gate:
    def __init__(self) -> None:
        self.tree = Path(os.environ["MB_GATE_WORKTREE"]).resolve()
        self.root = Path(os.environ["MB_GATE_ROOT"]).resolve()
        self.report = Path(os.environ["MB_GATE_REPORT"]).resolve()
        self.extension_seed = Path.home() / (
            "Library/Caches/mb/duckdb" if sys.platform == "darwin" else ".cache/mb/duckdb"
        )
        self.baseline = json.loads(
            (self.tree / "integration/gates/baseline.json").read_text(encoding="utf-8")
        )
        self.result = {
            "import": mainboard.__file__,
            "python": sys.executable,
            "temporary_root": str(self.root),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("pytest", "ruff", "pyrefly", "ty", "mypy", "mcmr")
            },
        }
        assert Path(mainboard.__file__).resolve() == self.tree / "src/mainboard/__init__.py"
        assert all(
            version == self.baseline["versions"][name]
            for name, version in self.result["versions"].items()
        ), "M00-VERSIONS: tool version changed; explicitly rebaseline before accepting results"
        sys.stdout.write(f"mainboard.__file__ = {mainboard.__file__}\n")

    def save(self, name: str) -> None:
        (self.report / f"{name}.json").write_text(
            json.dumps(self.result, indent=2) + "\n", encoding="utf-8"
        )

    def environment(self) -> dict[str, str]:
        environment = {
            key: os.environ[key]
            for key in ("PATH", "LANG", "SYSTEMROOT", "DYLD_FALLBACK_LIBRARY_PATH")
            if key in os.environ
        }
        for key, child in {
            "HOME": "home", "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache",
            "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data",
            "PIXI_CACHE_DIR": "cache/pixi", "UV_CACHE_DIR": "cache/uv",
            "MPLCONFIGDIR": "cache/matplotlib", "MYPY_CACHE_DIR": "cache/mypy",
        }.items():
            path = self.root / child
            path.mkdir(parents=True, exist_ok=True)
            environment[key] = str(path)
        environment.update(
            PYTHONPATH=f"{self.tree / 'src'}:{self.tree / 'integration/gates'}",
            PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", NO_COLOR="1",
            MB_GATE_ROOT=str(self.root), MB_GATE_REPORT=str(self.report),
            MB_GATE_WORKTREE=str(self.tree), MB_GATE_OS=sys.platform,
            MB_GATE_PROJECTS=os.environ["MB_GATE_PROJECTS"],
            MB_GATE_BOOTSTRAP="0",
        )
        return environment

    def sandbox(self, command: list[str]) -> list[str]:
        if sys.platform == "darwin":
            # Seatbelt restrictions are inherited by native libraries and all descendants.
            root = json.dumps(str(self.root))
            report = json.dumps(str(self.report))
            profile = self.root / "worker.sb"
            profile.write_text(
                '(version 1)\n(allow default)\n(deny file-write*)\n(deny network*)\n'
                f'(allow file-write* (subpath {root}) (subpath {report}) '
                '(literal "/dev/null"))\n'
                f'(allow network* (local unix-socket (subpath {root})) '
                f'(remote unix-socket (subpath {root})))\n', encoding="utf-8"
            )
            return ["/usr/bin/sandbox-exec", "-f", str(profile), *command]
        if sys.platform == "linux" and (bwrap := shutil.which("bwrap")):
            # A fresh root exposes runtime/source directories, not home, host sockets or devices.
            mounts = {
                Path(path).resolve() for path in (*sys.path, sys.base_prefix, str(self.tree))
                if path and Path(path).is_dir()
            }
            mounts.update(Path(path) for path in ("/usr", "/bin", "/lib", "/lib64", "/etc")
                          if Path(path).exists())
            bindings = [part for path in sorted(mounts) for part in
                        ("--ro-bind", str(path), str(path))]
            return [
                bwrap, "--die-with-parent", "--new-session", "--unshare-net", "--unshare-pid",
                "--tmpfs", "/", *bindings, "--tmpfs", "/tmp", "--tmpfs", "/run",
                "--dev", "/dev", "--proc", "/proc", "--bind", str(self.root), str(self.root),
                "--bind", str(self.report), str(self.report), *command,
            ]
        raise RuntimeError("M00-OS: needs macOS sandbox-exec or Linux bubblewrap")

    def execute(self, name: str, command: list[str], limit: int, *, isolated: bool = True) -> int:
        started = time.monotonic()
        environment = self.environment()
        with (self.report / f"{name}.log").open("w", encoding="utf-8") as output:
            child = subprocess.Popen(
                self.sandbox(command) if isolated else command, cwd=self.tree, env=environment,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                code = child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=3)
                code = 124
        self.result[name] = {
            "argv": command, "limit_seconds": limit, "exit_code": code,
            "seconds": round(time.monotonic() - started, 3),
        }
        return code

    def preflight(self) -> bool:
        # -S removes sitecustomize: these must be kernel refusals, not Python guard refusals.
        probe = self.tree / "integration/gates/native_probe.py"
        code = self.execute("isolation", [sys.executable, "-S", str(probe), str(self.root)], 20)
        if code:
            self.result["blocked"] = "M00-OS: OS isolation unavailable; no suite/checker launched"
            sys.stderr.write((self.report / "isolation.log").read_text(encoding="utf-8"))
        return code == 0

    def seed_extensions(self) -> None:
        # Copy preinstalled extensions; a gate never downloads code on demand.
        if not self.extension_seed.is_dir():
            raise RuntimeError(
                f"M00-EXTENSIONS: offline DuckDB cache missing: {self.extension_seed}"
            )
        cache = self.root / (
            "home/Library/Caches/mb/duckdb" if sys.platform == "darwin" else "cache/mb/duckdb"
        )
        shutil.copytree(self.extension_seed, cache, dirs_exist_ok=True)

    def inspect(self) -> int:
        # This lane never executes native test scenarios or subprocesses. It is not the gate.
        guard = Guard.from_environment()
        guard.deny_processes = True
        guard.install()
        environment = self.environment()
        os.environ.clear()
        os.environ.update(environment)
        os.environ["MB_GATE_INSPECT"] = "1"
        signal.setitimer(signal.ITIMER_REAL, 100)
        proofs = []
        for path in (
            Path(os.environ["MB_GATE_PROJECTS"]) / ".mainboard/M00-forbidden",
            self.tree / "M00-forbidden",
        ):
            with pytest.raises(PermissionError, match="M00 denies"):
                path.write_text("forbidden", encoding="utf-8")
            proofs.append(str(path))
        with pytest.raises(PermissionError, match="M00 denies real transport"):
            subprocess.run(["ssh", "gold", "true"], check=True)
        with pytest.raises(PermissionError, match="M00 denies DNS"):
            socket.create_connection(("api.vast.ai", 443), timeout=1)
        # The inspection-specific hook rejects every child, including transport attempts.
        self.result["python_denied_writes"] = proofs
        collector = Collection()
        self.result["collection_exit"] = pytest.main([
            "-c", str(self.tree / "pyproject.toml"), "-p", "no:randomly",
            "-p", "no:cacheprovider", "--collect-only", "-q", str(self.tree / "integration"),
        ], plugins=[collector])
        self.result["collected"] = collector.nodes
        original = {case["nodeid"] for case in self.baseline["cases"]}
        collected = set(collector.nodes)
        assert not original - collected, f"unaccounted baseline cases: {original - collected}"
        self.result["baseline_accounting"] = {
            "historical_counts": self.baseline["counts"],
            "original_collected": len(original & collected),
            "new_collected": len(collected - original),
            "missing": sorted(original - collected),
            "execution": "collection only; historical outcomes are not fresh gate passes",
        }
        binaries = Path(sys.executable).parent
        self.result["postgres"] = {
            "binaries": {name: str(binaries / name) for name in
                         ("postgres", "pg_ctl", "initdb", "psql")},
            "missing": [name for name in ("postgres", "pg_ctl", "initdb", "psql")
                        if not (binaries / name).is_file()],
            "lifecycle": "requires successful OS preflight; never start during inspection",
        }
        self.cli_witness()
        checks = sys.modules["integration.gates.test_harness"]
        scratch = self.root / "guard-check"
        scratch.mkdir()
        checks.test_live_paths_symlinks_and_transports_are_refused(scratch)
        (scratch / "escape").unlink()
        self.result["python_guard_witness"] = "passed, including safe symlink cleanup"
        signal.setitimer(signal.ITIMER_REAL, 0)
        return int(self.result["collection_exit"])

    def cli_witness(self) -> None:
        # Reuse the existing manifest -> parser -> Board -> plan -> JSON assertion in-process.
        # No native process is permitted in this limited inspection lane.
        from integration.conftest import Ran
        from integration.test_commands import (
            test_an_environment_narrowing_the_mirror_still_ships_every_lock_input,
        )
        from mainboard.cli import build

        workspace = self.root / "workspace"
        workspace.mkdir()

        def mb(*args: str) -> Ran:
            out, err = io.StringIO(), io.StringIO()
            with (
                contextlib.chdir(workspace), contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(err),
            ):
                try:
                    code = build()(args, exit_on_error=False)
                except SystemExit as exit_status:
                    code = exit_status.code
            return Ran(code=code or 0, out=out.getvalue(), err=err.getvalue())

        test_an_environment_narrowing_the_mirror_still_ships_every_lock_input(mb, workspace)
        self.result["cli_witness"] = {
            "nodeid": "test_an_environment_narrowing_the_mirror_still_ships_every_lock_input",
            "mode": "in-process, every subprocess denied; not a console-process gate pass",
            "result": "passed",
        }

    def run(self, mode: str, selection: tuple[str, ...]) -> int:
        if mode == "inspect":
            return self.inspect()
        if mode in {"suite", "target"}:
            if not self.preflight():
                return 78
            self.seed_extensions()
            targets = list(selection) or [str(self.tree / "integration")]
            command = [
                sys.executable, "-m", "pytest", "-c", str(self.tree / "pyproject.toml"),
                "-q", "-p", "no:randomly", "-p", "no:cacheprovider",
                "--basetemp", str(self.root / "pytest"),
                "--junitxml", str(self.report / f"{mode}.xml"), *targets,
            ]
            return self.execute(mode, command, 850 if mode == "suite" else 90)
        config = str(self.tree / "pyproject.toml")
        commands = {
            "ruff": ["ruff", "check", "--config", config, "--no-cache", "--output-format",
                     "json", "src", "integration"],
            "pyrefly": ["pyrefly", "check", "--config", config, "--min-severity", "warn",
                        "--output-format", "json"],
            "ty": ["ty", "check", "--project", str(self.tree), "--python", sys.executable,
                   "--output-format", "concise", "src"],
            "mypy": ["mypy", "--config-file", config, "--no-incremental",
                     "--show-error-codes", "--no-color-output", "src"],
            "mcmr": ["mcmr", "check", str(self.tree), "--deterministic", "--no-contextual",
                     "--no-external", "--no-writeback", "--repair", "none", "--format",
                     "concise", "--output", str(self.report / "mcmr-findings.json")],
        }
        if mode not in commands:
            raise ValueError(f"unknown gate mode: {mode}")
        # Fixed read-only checker invocations, not arbitrary commands or test scenarios.
        return self.execute(mode, commands[mode], 280, isolated=False)


class Collection:
    def __init__(self) -> None:
        self.nodes: list[str] = []

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        self.nodes = [item.nodeid for item in items]


def main(mode: str = "suite", *selection: str) -> None:
    gate = Gate()
    try:
        code = gate.run(mode, selection)
    finally:
        gate.save(mode)
    raise SystemExit(code)


if __name__ == "__main__":
    App(default_command=main)()
