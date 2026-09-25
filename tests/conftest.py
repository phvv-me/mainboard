import os
import shutil
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from types import SimpleNamespace

import pytest
from hypothesis import HealthCheck, settings

from mainboard import Board, ComputePath, HostFacts, Project
from mainboard.compute import Access, Survey
from mainboard.deps import Change, Dependencies
from mainboard.dispatch import Handle, HostSetup
from mainboard.dispatch.backends import Credentials
from mainboard.dispatch.dispatcher import Dispatcher
from mainboard.dispatch.shared import db_file
from mainboard.dispatch.state import Cache, DownHost, Failed, Finished, MonitorReport
from mainboard.doctor import Doctor, Section, Verdict
from mainboard.monitor import Monitor
from mainboard.scaffold import Scaffold, Scaffolded
from mainboard.verdicts import StreamVerdict, TrialVerdict, Verdicts

from .support import Answer, Lab, Launcher, Option, Owner, Relayed, build_lab

# The trials plugin ships as a pytest entry point, so the only honest way to test its hooks is to
# run pytest inside pytest, which is what `pytester` is for. It has to be named here because
# pytest reads `pytest_plugins` from the top-level conftest alone.
pytest_plugins = ["pytester", "mainboard.testing"]

# The stand-in for "this module was never imported", so the tracking seal restores absence as
# faithfully as it restores a module.
_ABSENT = object()

_MANIFEST = Project().manifest

# Hypothesis runs derandomized: the gate demands every line and branch on every run, so a property
# that reaches a branch must reach it again tomorrow, and with no example database a fresh checkout
# behaves like one that has run before. The budget is small for a fast inner loop;
# `--hypothesis-profile=deep` spends a much larger one when someone is hunting rather than gating.
_SHARED = {"deadline": None, "suppress_health_check": [HealthCheck.function_scoped_fixture]}
settings.register_profile("fast", derandomize=True, max_examples=30, **_SHARED)
settings.register_profile("deep", max_examples=500, **_SHARED)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))

_FIXTURE = """
[workspace]
name = "lab"
platforms = ["linux-64", "linux-aarch64"]

[vars]
cuda = "13.0"
scratch = "{{ env('MC_TEST_SCRATCH', '/tmp') }}"
station = "{{ os_name() }}-{{ arch() }}"

[deps]
python = ">=3.14"
pueue = "*"

[python.deps]
torch = ">=2.9"
lab-core = { path = "packages/lab-core", editable = true }

[envs.serving]
system = { cuda = "{{ vars.cuda }}" }

[envs.serving.python.deps]
vllm = "*"

[containers.ngc]
image = "nvcr.io/nvidia/pytorch:25.06-py3"
binds = ["{{ vars.scratch }}"]

[hosts.defaults]
sync = { include = ["packages"], protect = ["results/***"] }

[hosts.defaults.defaults]
walltime = "00:30:00"

[hosts.gold]
kind = "ssh"
env = "serving"

[hosts.miyabi-g]
kind = "pbs"
root = "/work/xg25g007/x10537/projects"
account = "xg25g007"
container = "ngc"
modules = { singularity = "4.2.1" }
scratch = "{{ env('LOCALDIR', '/local') }}"
sync = { exclude = ["data/raw"] }

[hosts.miyabi-g.queues.short-g]
max-walltime = "07:59:59"
mem-ceiling-gb = 100
gpus-per-node = 1

[hosts.miyabi-g.queues.debug-g]
max-walltime = "00:30:00"

[hosts.miyabi-g.defaults]
queue = "debug-g"
mem-gb = "min(100, attempt * 50)"

[tasks]
test = { run = "pytest", dir = "packages/lab-core" }

[tracking]
mode = "off"

[gates]
lint = "ruff check ."

[gates.proofs]
run = "prove doctor"
report = "result.breakages"
install = "mainboard add prove -l python"

[templates]
study = { path = "templates/study", into = "studies", answers = { home = "monorepo" } }
tool = "templates/tool"
"""


@pytest.fixture(autouse=True)
def sealed_tracking() -> Iterator[None]:
    """Keep every test off the real tracking SDK, whatever a manifest under test declares.

    Tracking is on by default, so a manifest silent about it would open real runs. Halting the
    import makes the sink refuse as on a machine without the package, which `Mirrored` absorbs
    like any refusal. A test wanting a sink patches its own stand-in later and so wins.

    The restore is by hand because asking a root autouse fixture for `monkeypatch` moves its
    teardown after every package conftest's, one of which clears caches a test had patched.
    """
    held = sys.modules.get("wandb", _ABSENT)
    sys.modules["wandb"] = None
    yield
    if held is _ABSENT:
        sys.modules.pop("wandb", None)
    else:
        sys.modules["wandb"] = held


@pytest.fixture(autouse=True)
def sealed_credentials() -> None:
    """Keep the developer's own workspace `.env` out of every test.

    The loader merges that file into the environment on the first key lookup, so a test clearing
    a key would watch it come back. Marked spent, the suite reads the same on a keyed machine as
    on a bare one; the loader's own tests unseal it onto a workspace they built.
    """
    Credentials().loaded = True


@pytest.fixture(scope="session")
def posix_bash() -> str:
    """A real POSIX Bash, never Windows' WSL launcher shim.

    GitHub's Windows image puts ``System32/bash.exe`` on PATH even with no WSL distribution, so
    Windows takes Git for Windows' Bash and coreutils explicitly rather than the ambiguous name.
    """
    if sys.platform != "win32":
        if bash := shutil.which("bash"):
            return bash
        pytest.skip("remote Bash protocol test needs Bash")

    roots = [Path(git).resolve().parent.parent] if (git := shutil.which("git")) else []
    roots.extend(
        Path(value) / "Git"
        for name in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)")
        if (value := os.environ.get(name))
    )
    if bash := next(
        (root / "bin" / "bash.exe" for root in roots if (root / "bin" / "bash.exe").is_file()),
        None,
    ):
        return str(bash)
    pytest.skip("remote Bash protocol test needs Git for Windows Bash")


@pytest.fixture(scope="session")
def lab_source(tmp_path_factory: pytest.TempPathFactory) -> Lab:
    """The lab built once: two `git init`s and a submodule add are the slow part of any test."""
    return build_lab(tmp_path_factory.mktemp("lab") / "projects")


@pytest.fixture
def lab(lab_source: Lab, tmp_path: Path) -> Iterator[Lab]:
    """A private copy of the lab to dirty, the modules the runner imported from it forgotten after.

    A copy keeps the submodule whole: its `.git` file points into the superproject's own
    `.git/modules`, which travels with the tree.
    """
    root = tmp_path / "projects"
    shutil.copytree(lab_source.root, root, symlinks=True)
    yield Lab(root)
    for name in [name for name in sys.modules if name.partition(".")[0] in _LAB_NAMES]:
        del sys.modules[name]
    sys.path[:] = [entry for entry in sys.path if not entry.startswith(str(root))]


# The top-level names the lab's packages import as, forgotten after every test that ran one.
_LAB_NAMES = frozenset({"experiments", "core", "sub"})


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace directory holding the full-featured fixture manifest."""
    monkeypatch.setenv("MC_TEST_SCRATCH", "/scratch/lab")
    (tmp_path / _MANIFEST).write_text(_FIXTURE)
    return tmp_path


# The shared dispatch database's tables, emptied together after any test that wrote to one.
_TABLES = ("runs", "hosts", "history")

# The kernel's tmpfs: the shared workspace below is mostly SQLite writes, each WAL commit one
# fsync that costs milliseconds on a real disk against microseconds here.
_MEMORY = Path("/dev/shm")

# The handle a recorded submit answers with, so a test reads a fixed id out of the rendered row.
_HANDLE = Handle(id="4242", host="miyabi-g", root="/work/p", kind="pbs")

# The task rows a rendered project leaves to paste, which `new` prints beside its record.
_SNIPPET = 'sc-baseline = { run = "python -m experiments.baseline.run execute" }\n'

# One dependency edit, the constraint that moved and the pin its solve dragged along.
_MOVED = [
    Change(name="tqdm", where="[dev.python.deps]", before="absent", after=">=4.70.0, <5"),
    Change(name="tqdm", where="pixi.lock", before="absent", after="4.70.0"),
]


def swept() -> MonitorReport:
    """A sweep report carrying one job of every outcome, so a render covers each row shape."""
    return MonitorReport(
        running=2,
        finished=[Finished(handle="1", target="gold", pulled_path="results/run")],
        failed=[Failed(handle="2", target="gold", reason="exited 137 (out of memory)")],
        unreachable_hosts=[DownHost(host="miyabi-g", reason="daemon down")],
    )


def settled() -> StreamVerdict:
    """One settled stream with a clean row, what the completion verbs render and exit on."""
    return StreamVerdict(
        stream="smoke-1",
        trials=(TrialVerdict(job="a", handle="4242", target="gold", verdict="ok", exit_code=0),),
    )


def surveyed() -> list[ComputePath]:
    """One row of every shape a compute table can hold, so a render covers each cell."""
    return [
        ComputePath(name="local", kind="local", access=Access.HERE, detail="1x RTX 4090, 64 GB"),
        ComputePath(name="miyabi-g", kind="pbs", access=Access.UNREACHABLE, detail="timed out"),
        ComputePath(
            name="vast",
            kind="provider",
            access=Access.KEYED,
            detail="1x RTX 4090 Texas, US",
            usd_hr=0.31,
            credit_usd=42.5,
        ),
    ]


@pytest.fixture(scope="session")
def station() -> Iterator[Path]:
    """The one workspace the root-level modules share, its dispatch database created once.

    Creating a SQLite file is fsync bound at tens of milliseconds and any board reaching for a
    dispatcher opens it, so a workspace per test paid that per test; `depot` empties it instead.
    """
    under = _MEMORY if os.access(_MEMORY, os.W_OK) else None
    root = Path(mkdtemp(dir=under, prefix="mainboard-station-"))
    (root / _MANIFEST).write_text(_FIXTURE)
    Cache(root / db_file()).connection.close()
    yield root
    rmtree(root, ignore_errors=True)


@pytest.fixture
def depot(station: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """`station` entered as the working directory, with whatever a test recorded dropped after.

    Rows are counted first since most tests write none, and even an empty `DELETE` syncs.
    """
    monkeypatch.setenv("MC_TEST_SCRATCH", "/scratch/lab")
    monkeypatch.chdir(station)
    yield station
    for generated in ("studies", "batches"):
        rmtree(station / Project().out_dir / generated, ignore_errors=True)
    cache = Cache(station / db_file())
    counted = " + ".join(f"(SELECT count(*) FROM {table})" for table in _TABLES)
    if cache.connection.execute(f"SELECT {counted} AS rows").fetchone()["rows"]:
        cache.connection.executescript(
            "BEGIN; " + " ".join(f"DELETE FROM {table};" for table in _TABLES) + " COMMIT;"
        )
    cache.connection.close()


@pytest.fixture
def board(depot: Path) -> Board:
    """A board over the shared station, reading the fixture manifest and that station's cache."""
    return Board(depot)


@pytest.fixture
def relayed(monkeypatch: pytest.MonkeyPatch) -> list[Relayed]:
    """Every board call a CLI verb makes, recorded as `(verb, host, args, options)` instead.

    Each stand-in answers with the shape the verb goes on to render, so the printing stays real
    while nothing behind the seam runs.
    """
    calls: list[Relayed] = []

    def relay(verb: str, answer: Answer) -> Callable[..., Answer]:
        def called(self: Owner, *args: str, **options: Option) -> Answer:
            watcher = options.pop("watch", None)
            calls.append((verb, getattr(self, "host", ""), args, options))
            if watcher is not None:
                watcher("probing")
            return answer

        return called

    made = Scaffolded(project="p", path="/p", snippet=_SNIPPET)
    for owner, verb, answer in (
        (Board, "run", 0),
        (Board, "submit", SimpleNamespace(handle=_HANDLE)),
        (Board, "install", HostSetup(host="gold", root="/repo", installer="uv")),
        (Board, "attest", None),
        (Board, "shell", None),
        (Board, "interact", None),
        (Board, "facts", HostFacts(schema_version=1, hostname="box")),
        (Board, "provide", Path("/envs/lab-4f2a")),
        (Dispatcher, "fetch_path", 3),
        (Dependencies, "add", _MOVED),
        (Dependencies, "remove", _MOVED),
        (Dependencies, "upgrade", _MOVED),
        (Scaffold, "render", made),
        (Doctor, "sections", [Section(section="fleet", verdict=Verdict.WARN, detail="asleep")]),
        (Survey, "paths", surveyed()),
        (Monitor, "once", swept()),
        (Monitor, "watch", iter([swept(), swept()])),
        (Verdicts, "cancel", settled()),
        (Verdicts, "captured", "the captured tail\n"),
        (Verdicts, "of", settled()),
        (Verdicts, "wait", settled()),
    ):
        monkeypatch.setattr(owner, verb, relay(verb, answer))
    return calls


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch) -> Launcher:
    """The lane collection subprocess replaced by a recorder that prints no cells until told to."""
    recorder = Launcher()
    monkeypatch.setattr("mainboard.cli.localhost", recorder)
    return recorder
