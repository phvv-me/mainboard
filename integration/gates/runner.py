"""Run one gate lane inside the bounded, OS-isolated worker that run.sh launches."""

import contextlib
import importlib.metadata
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import pytest
from cyclopts import App
from guard import Guard

import mainboard

# Seconds of the budget that launching, preflight and cleanup keep for themselves.
_RESERVE = 30
_TOOLS = ["pytest", "ruff", "pyrefly", "ty", "mypy", "mcmr"]
_INHERITED = ["PATH", "LANG", "SYSTEMROOT", "DYLD_FALLBACK_LIBRARY_PATH", "DEVELOPER_DIR"]
_HOMES = {
    "HOME": "home",
    "TMPDIR": "tmp",
    "XDG_CACHE_HOME": "cache",
    "XDG_CONFIG_HOME": "config",
    "XDG_DATA_HOME": "data",
    "PIXI_CACHE_DIR": "cache/pixi",
    "UV_CACHE_DIR": "cache/uv",
    "MPLCONFIGDIR": "cache/matplotlib",
    "MYPY_CACHE_DIR": "cache/mypy",
}
_POSTGRES = ["postgres", "pg_ctl", "initdb", "psql"]
# A lane holds only the tool it runs to that tool's recorded version: pytest for the lanes that
# run it, else the checker the lane is named after, and none for `seed`.
_TESTERS = {"suite", "target", "inspect"}
# The DuckDB extensions `mainboard.state.lake` loads to attach a lake and to serve or reach one
# over Quack, and the user cache every sandbox copies them from.
_EXTENSIONS = ("ducklake", "sqlite", "httpfs", "quack")
_SEED = Path.home() / ("Library/Caches" if sys.platform == "darwin" else ".cache") / "mb/duckdb"
# What the inspect lane runs in this process, where every child process is refused.
_HARNESS = "integration/gates/test_harness.py"
_WITNESSES = [
    f"{_HARNESS}::test_a_write_outside_the_roots_is_refused",
    f"{_HARNESS}::test_a_provider_name_is_refused",
    f"{_HARNESS}::test_the_installed_hook_runs_first",
    "integration/test_commands.py::"
    "test_an_environment_narrowing_the_mirror_still_ships_every_lock_input",
]


class Gate:
    """One lane of the gate: its worker's environment, sandbox, bounds and report."""

    def __init__(self, mode: str) -> None:
        self.tree = Path(os.environ["MB_GATE_WORKTREE"]).resolve()
        self.root = Path(os.environ["MB_GATE_ROOT"]).resolve()
        self.report = Path(os.environ["MB_GATE_REPORT"]).resolve()
        self.budget = int(os.environ["MB_GATE_BUDGET"]) - _RESERVE
        self.started = time.monotonic()
        self.baseline = json.loads(
            (self.tree / "integration/gates/baseline.json").read_text(encoding="utf-8")
        )
        tool = "pytest" if mode in _TESTERS else mode
        versions = {name: importlib.metadata.version(name) for name in _TOOLS if name == tool}
        self.result = {
            "import": mainboard.__file__,
            "python": sys.executable,
            "temporary_root": str(self.root),
            "budget_seconds": self.budget + _RESERVE,
            "versions": versions,
        }
        self._verify(versions)

    def run(self, mode: str, selection: list[str]) -> int:
        """Run lane `mode` and return the exit code its worker ended with."""
        if mode == "inspect":
            return self._inspect()
        if mode in {"suite", "target"}:
            return self._pytest(mode, selection)
        if mode == "seed":
            return self._install()
        return self._execute(mode, self._checker(mode), self.budget, isolated=False)

    def save(self, name: str) -> None:
        self.result["wall_seconds"] = round(time.monotonic() - self.started, 3)
        (self.report / f"{name}.json").write_text(
            json.dumps(self.result, indent=2) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _wait(child: subprocess.Popen[bytes], limit: int) -> int:
        """The exit code of `child`, or 124 once `limit` passed and its group was ended."""
        with contextlib.suppress(subprocess.TimeoutExpired):
            return child.wait(timeout=limit)
        os.killpg(child.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            child.wait(timeout=3)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        return 124

    def _accounting(self, listing: str) -> dict[str, object]:
        """Every baseline case must still be collected; additions are listed apart."""
        collected = {line for line in listing.splitlines() if "::" in line}
        original = {case["nodeid"] for case in self.baseline["cases"]}
        assert original <= collected, f"unaccounted baseline cases: {sorted(original - collected)}"
        return {
            "historical_counts": self.baseline["counts"],
            "original_collected": len(original),
            "new_collected": len(collected - original),
            "collected": sorted(collected),
            "execution": "collection only; historical outcomes are not fresh gate passes",
        }

    def _arguments(self) -> list[str]:
        return [
            "-c",
            str(self.tree / "pyproject.toml"),
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(self.root / "pytest"),
        ]

    def _bubblewrap(self, bwrap: str, command: list[str]) -> list[str]:
        """A fresh root exposing runtime and source directories, never home, sockets or devices."""
        runtime = [*sys.path, sys.base_prefix, str(self.tree)]
        mounts = {Path(path).resolve() for path in runtime if path and Path(path).is_dir()}
        system = [Path(path) for path in ["/usr", "/bin", "/lib", "/lib64", "/etc"]]
        mounts.update(path for path in system if path.exists())
        bindings = [
            part for mount in sorted(mounts) for part in ("--ro-bind", str(mount), str(mount))
        ]
        executable = Path(sys.executable)
        target = executable.parent / os.readlink(executable) if executable.is_symlink() else None
        # uv's `bin/python` names its interpreter through a directory that is itself a symlink to
        # the versioned one, and only the real directory is mounted above.
        links = [link for link in reversed(target.parents) if link.is_symlink()] if target else []
        symlinks = [part for link in links for part in ("--symlink", os.readlink(link), str(link))]
        return [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--unshare-pid",
            "--tmpfs",
            "/",
            *bindings,
            *symlinks,
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/run",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--bind",
            str(self.root),
            str(self.root),
            "--bind",
            str(self.report),
            str(self.report),
            *command,
        ]

    def _checker(self, mode: str) -> list[str]:
        """A fixed, read-only checker invocation: never an arbitrary command or test scenario."""
        config = str(self.tree / "pyproject.toml")
        findings = self.report / "mcmr-findings.json"
        return {
            "ruff": [
                "ruff",
                "check",
                "--config",
                config,
                "--no-cache",
                "--output-format",
                "json",
                "src",
                "integration",
            ],
            "pyrefly": [
                "pyrefly",
                "check",
                "--config",
                config,
                "--min-severity",
                "warn",
                "--output-format",
                "json",
            ],
            "ty": [
                "ty",
                "check",
                "--project",
                str(self.tree),
                "--python",
                sys.executable,
                "--output-format",
                "concise",
                "src",
            ],
            "mypy": [
                "mypy",
                "--config-file",
                config,
                "--no-incremental",
                "--show-error-codes",
                "--no-color-output",
                "src",
            ],
            "mcmr": [
                "mcmr",
                "check",
                str(self.tree),
                "--deterministic",
                "--no-contextual",
                "--no-external",
                "--no-writeback",
                "--repair",
                "none",
                "--format",
                "concise",
                "--output",
                str(findings),
            ],
        }[mode]

    def _collect(self) -> tuple[int, str]:
        """pytest's exit code and node listing for the whole suite, without running a test."""
        listing = io.StringIO()
        arguments = [*self._arguments(), "--collect-only", str(self.tree / "integration")]
        with contextlib.redirect_stdout(listing):
            code = pytest.main(arguments)
        return int(code), listing.getvalue()

    def _environment(self) -> dict[str, str]:
        """The worker's environment, rebuilt from an allowlist, every state path in the root."""
        environment = {key: os.environ[key] for key in _INHERITED if key in os.environ}
        for key, child in _HOMES.items():
            (self.root / child).mkdir(parents=True, exist_ok=True)
            environment[key] = str(self.root / child)
        environment.update(
            PYTHONPATH=f"{self.tree / 'src'}:{self.tree / 'integration/gates/worker'}",
            PYTHONDONTWRITEBYTECODE="1",
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            NO_COLOR="1",
            MB_GATE_ROOT=str(self.root),
            MB_GATE_REPORT=str(self.report),
            MB_GATE_WORKTREE=str(self.tree),
            MB_GATE_OS=sys.platform,
            MB_GATE_PROJECTS=os.environ["MB_GATE_PROJECTS"],
        )
        return environment

    def _execute(self, name: str, command: list[str], limit: int, *, isolated: bool = True) -> int:
        """Run `command` in its own process group, ended whole once `limit` seconds pass."""
        started = time.monotonic()
        with (self.report / f"{name}.log").open("w", encoding="utf-8") as output:
            child = subprocess.Popen(
                self._sandbox(command) if isolated else command,
                cwd=self.root if isolated else self.tree,
                env=self._environment(),
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            code = self._wait(child, limit)
        self.result[name] = {
            "argv": command,
            "limit_seconds": limit,
            "exit_code": code,
            "seconds": round(time.monotonic() - started, 3),
        }
        return code

    def _install(self) -> int:
        """Install the extensions a sandbox is seeded from, the one lane that goes online.

        A machine that never ran the tool needs it once; installing what is there changes nothing.
        """
        connection = duckdb.connect()
        connection.execute(f"SET extension_directory = '{_SEED.as_posix()}'")
        for name in _EXTENSIONS:
            connection.install_extension(name)
        self.result["seed"] = {"directory": str(_SEED), "extensions": _EXTENSIONS}
        return 0

    def _inspect(self) -> int:
        """Collect, then run the reviewed witnesses here with every child process refused.

        This lane has no OS isolation, so a pass here is not a suite pass.
        """
        self._refuse_children()
        code, listing = self._collect()
        witnesses = pytest.main([*self._arguments(), *(str(self.tree / n) for n in _WITNESSES)])
        binaries = Path(sys.executable).parent
        self.result.update(
            collection_exit=code,
            baseline_accounting=self._accounting(listing),
            witnesses={"nodes": _WITNESSES, "exit_code": int(witnesses)},
            postgres={"missing": [n for n in _POSTGRES if not (binaries / n).is_file()]},
        )
        return max(code, int(witnesses))

    def _preflight(self) -> bool:
        """Prove the OS refuses an external write and connection before any test code runs."""
        probe = self.tree / "integration/gates/native_probe.py"
        code = self._execute("isolation", [sys.executable, "-S", str(probe), str(self.root)], 20)
        if code:
            self.result["blocked"] = "M00-OS: OS isolation unavailable; no suite/checker launched"
            sys.stderr.write((self.report / "isolation.log").read_text(encoding="utf-8"))
        return code == 0

    def _pytest(self, mode: str, selection: list[str]) -> int:
        if not self._preflight():
            return 78
        self._seed_extensions()
        command = [
            sys.executable,
            "-m",
            "pytest",
            *self._arguments(),
            "--junitxml",
            str(self.report / f"{mode}.xml"),
            *(str(self.tree / node) for node in selection or ["integration"]),
        ]
        return self._execute(mode, command, self.budget)

    def _refuse_children(self) -> None:
        guard = Guard.from_environment()
        guard.deny_processes = True
        guard.install()
        environment = self._environment()
        os.environ.clear()
        os.environ.update(environment, MB_GATE_INSPECT="1")
        signal.setitimer(signal.ITIMER_REAL, self.budget)

    def _sandbox(self, command: list[str]) -> list[str]:
        if sys.platform == "darwin":
            return self._seatbelt(command)
        if sys.platform == "linux" and (bwrap := shutil.which("bwrap")):
            return self._bubblewrap(bwrap, command)
        raise RuntimeError("M00-OS: needs macOS sandbox-exec or Linux bubblewrap")

    def _seatbelt(self, command: list[str]) -> list[str]:
        """Wrap `command` in a Seatbelt profile that refuses outside writes and the network.

        Only the root, the report, /dev/null and ptys are writable, and only Unix sockets below
        the root are reachable. Native libraries and every descendant inherit the restriction.
        """
        root, report = json.dumps(str(self.root)), json.dumps(str(self.report))
        profile = self.root / "worker.sb"
        profile.write_text(
            "(version 1)\n(allow default)\n(deny file-write*)\n(deny network*)\n"
            f'(allow file-write* (subpath {root}) (subpath {report}) (literal "/dev/null") '
            '(literal "/dev/ptmx") (regex #"^/dev/ttys[0-9]+$"))\n'
            f"(allow network* (local unix-socket (subpath {root})) "
            f"(remote unix-socket (subpath {root})))\n",
            encoding="utf-8",
        )
        return ["/usr/bin/sandbox-exec", "-f", str(profile), *command]

    def _seed_extensions(self) -> None:
        """Copy the preinstalled DuckDB extensions; a gate never downloads code on demand."""
        if not _SEED.is_dir():
            raise RuntimeError(
                f"M00-EXTENSIONS: offline DuckDB cache missing: {_SEED}; run `run.sh seed` first"
            )
        darwin = sys.platform == "darwin"
        cache = self.root / ("home/Library/Caches/mb/duckdb" if darwin else "cache/mb/duckdb")
        shutil.copytree(_SEED, cache, dirs_exist_ok=True)
        # The AST provider uses DuckDB's standard extension tree, not the lake's custom cache.
        # Copy installed code only; no user database, profile or configuration enters the worker.
        extensions = Path.home() / ".duckdb/extensions"
        if extensions.is_dir():
            shutil.copytree(extensions, self.root / "home/.duckdb/extensions", dirs_exist_ok=True)

    def _verify(self, versions: dict[str, str]) -> None:
        """Refuse a worktree other than this one and a lane tool that drifted from the baseline."""
        assert Path(mainboard.__file__).resolve() == self.tree / "src/mainboard/__init__.py"
        assert versions == {name: self.baseline["versions"][name] for name in versions}, (
            "M00-VERSIONS: tool version changed; explicitly rebaseline before accepting results"
        )
        sys.stdout.write(f"mainboard.__file__ = {mainboard.__file__}\n")


def _main(mode: str = "suite", *selection: str) -> None:
    """Run lane `mode` (suite, target, inspect, seed or a checker); `target` takes node ids."""
    gate = Gate(mode)
    try:
        code = gate.run(mode, list(selection))
    finally:
        gate.save(mode)
    raise SystemExit(code)


if __name__ == "__main__":
    App(default_command=_main)()
