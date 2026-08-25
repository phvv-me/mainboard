import os
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
from mainboard.dispatch.shared import db_file
from mainboard.dispatch.state import Cache, DownHost, Failed, Finished, MonitorReport
from mainboard.doctor import Doctor, Section, Verdict
from mainboard.monitor import Monitor
from mainboard.scaffold import Scaffold, Scaffolded
from mainboard.verdicts import StreamVerdict, TrialVerdict, Verdicts

from .support import Answer, Option, Owner, Relayed

# The stand-in for "this module was never imported", so the tracking seal restores absence as
# faithfully as it restores a module.
_ABSENT = object()

_MANIFEST = Project().manifest

# Hypothesis runs derandomized here, which buys two things this suite needs. The gate demands
# every line and branch on every run, so a property that reaches a branch has to reach it again
# tomorrow, and a fixed example set is what makes that true. It also drops the example database,
# so a checkout with no `.hypothesis` directory behaves exactly like one that has run before.
# The default budget is small because the suite is a fast inner loop; `--hypothesis-profile=deep`
# spends a much larger one when someone is hunting rather than gating.
_SHARED = {
    "deadline": None,
    "suppress_health_check": [HealthCheck.function_scoped_fixture],
}
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

    Tracking is on by default, so a workspace fixture that says nothing about it would open real
    runs from inside the suite. Halting the import is the tightest seal available: the sink
    refuses exactly as it does on a machine that never installed the package, `Mirrored` absorbs
    that refusal exactly as it absorbs any other, and no test can reach the network by
    forgetting a table. A test that wants a sink installs its own stand-in over this one and
    wins, since it patches later and is undone first.

    The save and restore is written by hand rather than through `monkeypatch`, because asking a
    root autouse fixture for `monkeypatch` moves that fixture's teardown after every package
    conftest's, and one of them clears caches a test had patched.
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

    The credential loader merges that file into the process environment the first time a backend
    looks a key up, so a suite running inside a real workspace would inherit whatever keys the
    machine holds and a test that clears one would watch it come straight back. Marking the
    shared loader spent before each test makes the whole suite read the same on a keyed machine
    as on a bare one, and the loader's own tests unseal it onto a workspace they built.
    """
    Credentials().loaded = True


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace directory holding the full-featured fixture manifest."""
    monkeypatch.setenv("MC_TEST_SCRATCH", "/scratch/lab")
    (tmp_path / _MANIFEST).write_text(_FIXTURE)
    return tmp_path


# The three tables the shared dispatch database keeps, emptied together after any test that
# wrote to one of them.
_TABLES = ("runs", "hosts", "history")

# The tmpfs the kernel already offers. The shared workspace below is a SQLite file being written
# to and little else, every WAL commit is one fsync, and that fsync costs milliseconds on a real
# disk against microseconds here. A directory made under it is as hermetic as any other.
_MEMORY = Path("/dev/shm")

# The handle a recorded submit answers with, so a test reads a fixed id out of the rendered row.
_HANDLE = Handle(id="4242", host="miyabi-g", root="/work/p", kind="pbs")

# The task rows a rendered project leaves for the workspace to paste, the one payload the `new`
# verb prints beside its record rather than inside it.
_SNIPPET = 'sc-baseline = { run = "python -m experiments.baseline.run execute" }\n'

# One dependency edit as the curator reports it, the constraint that moved and the pin its solve
# dragged along, which is the two-row shape every dependency verb renders.
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

    That database was this slice's whole cost. A board opens it the moment anything reaches for
    a dispatcher, and creating a fresh SQLite file is fsync bound at tens of milliseconds, so a
    workspace per test paid for one per test. Nothing in the file is test-specific, so it is
    built once here and emptied after whichever test wrote to it.
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

    The row count is asked for before anything is deleted because most tests write nothing at
    all, and a `DELETE` that finds no rows still opens a transaction the shared file must sync.
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

    The verbs are a dispatch table over the board, so what belongs to the CLI is which method
    each one reaches and what it turned its flags into. Everything past that seam is tested
    where it lives. Each stand-in answers with the shape the verb goes on to render, so the
    printing stays real while nothing behind the seam runs.
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
        (Board, "shell", None),
        (Board, "interact", None),
        (Board, "facts", HostFacts(schema_version=1, hostname="box")),
        (Dependencies, "add", _MOVED),
        (Dependencies, "remove", _MOVED),
        (Dependencies, "upgrade", _MOVED),
        (Scaffold, "render", made),
        (Doctor, "sections", [Section(section="fleet", verdict=Verdict.WARN, detail="asleep")]),
        (Survey, "paths", surveyed()),
        (Monitor, "once", swept()),
        (Monitor, "watch", iter([swept(), swept()])),
        (Verdicts, "of", settled()),
        (Verdicts, "wait", settled()),
    ):
        monkeypatch.setattr(owner, verb, relay(verb, answer))
    return calls
