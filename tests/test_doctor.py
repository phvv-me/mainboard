import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from plumbum.commands.processes import ProcessTimedOut

from mainboard import Board, ComputePath, HostFacts, Survey
from mainboard.compute import Access
from mainboard.dispatch import HostSetup
from mainboard.doctor import Doctor, Section, Verdict
from mainboard.engines.compile import Provisioner
from mainboard.engines.compile.backend import CommandResult
from mainboard.engines.compile.state import SyncState
from mainboard.staleness import Snapshot

from .strategies import WORDS

_FINGERPRINT = ".pixi-environment-fingerprint"

# The two gates the fixture workspace declares: one that reports where its failures live, and one
# plain command that speaks nothing but its exit status.
_REPORTING = "proofs"
_BARE = "lint"

# A gate's own report as a tool stamps one, cut down to the one field the manifest points at.
_BROKEN = json.dumps(
    {"result": {"breakages": [".: failing_claims", "research/x: stale_claims"]}, "exit_status": 1}
)
_SETTLED = json.dumps({"result": {"breakages": []}, "exit_status": 0})

# The generated tree in every state the environment section tells apart, each step the one the
# step before it makes sense on, so a case names its state and the ladder walks up to it.
_LADDER = ("bare", "compiled", "solved", "provisioned", "blessed", "whole", "damaged")

# The compute paths a fleet section is handed, one row of any shape at all.
_ROWS = st.builds(ComputePath, name=WORDS, kind=WORDS, access=st.sampled_from(Access))

# What the section counts as usable as it stands, so anything outside it earns a word.
_USABLE = {Access.HERE, Access.READY, Access.KEYED}


def answering(status: int, output: str) -> Callable[[str, float], tuple[int, str]]:
    """A gate probe answering every command with one fixed exit status and output."""
    return lambda command, timeout: (status, output)


class FixedSurvey(Survey):
    """A survey answering with a fixed set of rows instead of reaching for a machine."""

    def __init__(self, board: Board, rows: Sequence[ComputePath]) -> None:
        super().__init__(board, providers=[])
        self.rows = list(rows)

    def paths(self, setups: Mapping[str, HostSetup] | None = None) -> list[ComputePath]:
        del setups
        return self.rows


def climbed(workspace: Path, stage: str) -> None:
    """Walk the workspace's generated tree up to `stage` and stop there.

    bare is a workspace nothing ever compiled. compiled has a manifest and a lock no digest
    vouches for. solved records the current digests without an installed prefix. provisioned
    installs `default` without recording what it was built from, blessed records those digests,
    whole does the same for `serving`, and damaged then takes an installed wheel's import roots
    away underneath pixi.
    """
    if stage == "bare":
        return
    provisioner = Provisioner(workspace, Board(workspace).manifest)
    provisioner.out.mkdir(parents=True, exist_ok=True)

    def compile(env: str) -> None:
        directory = provisioner.environment_dir(env)
        directory.mkdir(parents=True, exist_ok=True)
        pixi = provisioner.pixi_for(env)
        pixi.manifest.write_text('[workspace]\nname = "lab"\n', encoding="utf-8")
        pixi.lock.write_text("version: 6\n", encoding="utf-8")
        SyncState.path(directory).write_text(SyncState().render(), encoding="utf-8")

    compile("default")
    if stage == "compiled":
        return

    def install(env: str) -> None:
        prefix = provisioner.pixi_for(env).env_prefix(env)
        (prefix / "lib" / "python3.14" / "site-packages").mkdir(parents=True, exist_ok=True)
        (prefix / "conda-meta").mkdir(parents=True, exist_ok=True)
        (prefix / "conda-meta" / _FINGERPRINT).write_text("installed\n", encoding="utf-8")

    def bless(env: str) -> None:
        directory = provisioner.environment_dir(env)
        compiler = provisioner.compiler_for(env)
        state = SyncState.load(directory)
        current = state.model_copy(
            update={
                "environment": env,
                "compiled_from": compiler.digest(),
                "solved_from": compiler.resolution_digest(),
            }
        )
        SyncState.path(directory).write_text(current.render(), encoding="utf-8")

    if stage == "solved":
        bless("default")
        return
    install("default")
    if stage == "provisioned":
        return
    bless("default")
    if stage == "blessed":
        return
    compile("serving")
    install("serving")
    bless("serving")
    if stage == "whole":
        return
    metadata = (
        provisioner.pixi_for("default").env_prefix("default")
        / "lib"
        / "python3.14"
        / "site-packages"
        / "ghost-1.0.dist-info"
    )
    metadata.mkdir(parents=True)
    metadata.joinpath("METADATA").write_text("Name: ghost\nVersion: 1.0\n")
    metadata.joinpath("INSTALLER").write_text("uv-pixi")
    metadata.joinpath("top_level.txt").write_text("ghost\n")


def test_a_manifest_that_loads_reports_what_it_declares(workspace: Path) -> None:
    """The pass line is the workspace's own shape, which is what a reader wants confirmed."""
    found = Doctor(Board(workspace)).manifest()
    assert found.verdict is Verdict.PASS
    assert "lab" in found.detail
    assert "2 environments" in found.detail


def test_a_manifest_that_will_not_load_is_the_whole_report(workspace: Path) -> None:
    """Every other section was going to read that manifest, so inventing them says nothing."""
    (workspace / "mainboard.toml").write_text("[workspace]\nname = 3\n")
    sections = Doctor(Board(workspace)).sections()
    assert [found.section for found in sections] == ["manifest"]
    assert sections[0].verdict is Verdict.FAIL
    assert sections[0].fix == "mainboard check"


@pytest.mark.parametrize(
    ("stage", "verdict", "fragment", "fix"),
    [
        (
            "bare",
            Verdict.WARN,
            "default: nothing compiled yet",
            "mainboard install default --resolve",
        ),
        (
            "compiled",
            Verdict.FAIL,
            "pixi.lock was not solved from this manifest",
            "mainboard install default --resolve",
        ),
        ("solved", Verdict.WARN, "never installed: default", "mainboard install default"),
        (
            "provisioned",
            Verdict.FAIL,
            "compiled before the current manifest: default",
            "mainboard install default --resolve",
        ),
        ("blessed", Verdict.PASS, "default is provisioned, fresh and whole", ""),
        ("whole", Verdict.PASS, "default is provisioned, fresh and whole", ""),
        ("damaged", Verdict.FAIL, "needs reinstalling: ghost", "mainboard install default"),
    ],
    ids=[
        "nothing was ever built here, a first step and not a broken state",
        "a lock nothing on this disk vouches for is a broken workspace",
        "a solved environment still needs its first installation",
        "what is installed is not what the manifest describes",
        "installing what you need is normal, so the unbuilt ones are a word",
        "every declared environment provisioned, current and importable",
        "a wheel that lost its import roots, which no lock ever notices",
    ],
)
def test_the_environment_section_tells_apart_every_way_a_workspace_drifts(
    workspace: Path, stage: str, verdict: Verdict, fragment: str, fix: str
) -> None:
    climbed(workspace, stage)
    found = Doctor(Board(workspace)).environment()
    assert found.verdict is verdict
    assert fragment in found.detail
    assert found.fix == fix


@pytest.mark.parametrize(
    ("stage", "verdict", "fragment", "fix"),
    [
        (
            "blessed",
            Verdict.WARN,
            "serving: nothing compiled yet",
            "mainboard install serving --resolve",
        ),
        ("whole", Verdict.PASS, "serving is provisioned, fresh and whole", ""),
    ],
)
def test_the_environment_section_audits_only_the_selected_shard(
    workspace: Path, stage: str, verdict: Verdict, fragment: str, fix: str
) -> None:
    climbed(workspace, stage)
    found = Doctor(Board(workspace), env="serving").environment()
    assert (found.verdict, found.fix) == (verdict, fix)
    assert fragment in found.detail


@given(rows=st.lists(_ROWS, max_size=6))
@example(rows=[])
@example(rows=[ComputePath(name="gold", kind="ssh", access=Access.REACHABLE)])
@example(rows=[ComputePath(name="vast", kind="provider", access=Access.UNKEYED)])
@example(rows=[ComputePath(name="local", kind="local", access=Access.HERE)])
def test_the_fleet_verdict_is_a_pure_function_of_the_rows_it_was_handed(
    workspace: Path, rows: list[ComputePath]
) -> None:
    """The doctor reports the world and never fails on it.

    A sleeping host and an unkeyed provider are facts about the world rather than about this
    workspace, and the one repair offered is always the next thing a reader would type.
    """
    board = Board(workspace)
    found = Doctor(board, survey=FixedSurvey(board, rows)).fleet()
    usable = [row for row in rows if row.access in _USABLE]
    cold = [row.name for row in rows if row.access is Access.REACHABLE]
    assert found.verdict is not Verdict.FAIL
    assert (found.verdict is Verdict.PASS) == (len(usable) == len(rows))
    assert (found.fix == "") == (found.verdict is Verdict.PASS)
    assert found.detail.startswith(f"{len(usable)} ")
    if cold:
        assert found.fix == f"mainboard setup {cold[0]}"
    elif found.verdict is Verdict.WARN:
        assert found.fix == "mainboard compute"


def test_the_hosts_verdict_compares_each_recorded_digest_against_the_manifest_now(
    workspace: Path,
) -> None:
    """A host is provisioned once and the manifest can move any number of times after that."""
    board = Board(workspace)
    provisioner = Provisioner(board.root, board.manifest)
    current = provisioner.compiler_for("default").digest()
    serving = provisioner.compiler_for("serving").digest()
    fresh = HostSetup(host="gold", root="/repo", digest=current)
    fresh_serving = HostSetup(host="serve", root="/repo3", env="serving", digest=serving)
    stale = HostSetup(host="miyabi-g", root="/repo2", digest="not-the-current-digest")
    unrecorded = HostSetup(host="vast", root="/repo3")
    missing_environment = HostSetup(host="lost", root="/repo4", env="gone", digest="recorded")
    doctor = Doctor(board)

    agreeing = doctor.hosts({"gold": fresh, "serve": fresh_serving, "vast": unrecorded})
    assert agreeing.verdict is Verdict.PASS
    assert agreeing.detail == "3 onboarded, none diverged from the current manifest"
    assert agreeing.fix == ""

    diverged = doctor.hosts({"gold": fresh, "miyabi-g": stale, "vast": unrecorded})
    assert diverged.verdict is Verdict.WARN
    assert diverged.detail == "diverged from the current manifest: miyabi-g"
    assert diverged.fix == "mainboard setup miyabi-g --sync-only"

    missing = doctor.hosts({"lost": missing_environment})
    assert missing.verdict is Verdict.WARN
    assert missing.detail == "diverged from the current manifest: lost"
    assert missing.fix == "mainboard setup lost --sync-only"


@pytest.mark.parametrize(
    ("gate", "status", "output", "verdict", "detail", "fix"),
    [
        (
            _REPORTING,
            1,
            _BROKEN,
            Verdict.FAIL,
            "2 breakages: .: failing_claims, research/x: stale_claims",
            "mainboard run -- prove doctor",
        ),
        (
            _REPORTING,
            0,
            _SETTLED,
            Verdict.PASS,
            "`prove doctor` reports nothing broken",
            "",
        ),
        (
            _REPORTING,
            127,
            "prove: command not found\n",
            Verdict.WARN,
            "`prove doctor` exited 127 without a report, is it installed",
            "mainboard add prove -l python",
        ),
        (_REPORTING, 1, "", Verdict.WARN, "without a report", "mainboard add prove -l python"),
        (
            _REPORTING,
            1,
            "{not json",
            Verdict.WARN,
            "without a report",
            "mainboard add prove -l python",
        ),
        (
            _REPORTING,
            1,
            '{"result": 3}',
            Verdict.WARN,
            "without a report",
            "mainboard add prove -l python",
        ),
        (
            _REPORTING,
            1,
            '{"result": {"breakages": 3}}',
            Verdict.WARN,
            "without a report",
            "mainboard add prove -l python",
        ),
        (
            _BARE,
            1,
            "checking\nfound 3 errors\n",
            Verdict.FAIL,
            "found 3 errors",
            "mainboard run -- ruff check .",
        ),
        (
            _BARE,
            2,
            "  \n",
            Verdict.FAIL,
            "`ruff check .` exited 2",
            "mainboard run -- ruff check .",
        ),
    ],
    ids=[
        "the findings are the gate's own judgment and are passed on as written",
        "a clean exit is the gate saying everything it watches is settled",
        "a tool nobody installed cannot call this workspace broken",
        "no output at all is a gate that never ran",
        "output that is not a report is a gate that never ran",
        "a report of the wrong shape is a gate that never ran",
        "a breakage field of the wrong type is a gate that never ran",
        "a plain command complains on its last line",
        "a silent command is still named by its exit status",
    ],
)
def test_a_gate_is_told_apart_by_what_it_reported_as_well_as_by_its_exit(
    workspace: Path,
    gate: str,
    status: int,
    output: str,
    verdict: Verdict,
    detail: str,
    fix: str,
) -> None:
    found = Doctor(Board(workspace), probe=answering(status, output)).gate(gate)
    assert found.verdict is verdict
    assert detail in found.detail
    assert found.fix == fix


def test_a_gate_that_will_not_answer_in_time_is_a_word(workspace: Path) -> None:
    """The probe is bounded, and hitting that bound is reported rather than waited out."""

    def hang(command: str, timeout: float) -> tuple[int, str]:
        raise ProcessTimedOut("expired", ["prove"])

    found = Doctor(Board(workspace), probe=hang).gate(_REPORTING)
    assert found.verdict is Verdict.WARN
    assert "did not answer within 90s" in found.detail


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        ("", ["manifest", "environment", "snapshot", "fleet", "hosts", _BARE, _REPORTING]),
        (
            '[workspace]\nname = "bare"\n',
            ["manifest", "environment", "snapshot", "fleet", "hosts"],
        ),
    ],
    ids=["every gate the workspace declares", "a workspace that declares none"],
)
def test_the_sections_are_the_questions_asked_before_starting_work(
    workspace: Path, manifest: str, expected: list[str]
) -> None:
    """A section about a question nobody asked here would be a line that says nothing."""
    if manifest:
        (workspace / "mainboard.toml").write_text(manifest)
    board = Board(workspace)
    doctor = Doctor(board, survey=FixedSurvey(board, []), probe=answering(0, _SETTLED))
    sections = doctor.sections()
    assert [found.section for found in sections] == expected
    assert all(isinstance(found, Section) for found in sections)


def test_the_report_never_hands_the_dispatch_cache_to_a_thread_that_does_not_own_it(
    workspace: Path,
) -> None:
    """One SQLite connection, opened and used on the thread the report was asked from.

    The fleet section reads the onboarding records, and reaching for them from inside the pool
    opened the shared cache there, which left the interpreter closing a connection from a thread
    that never owned it and ended a whole clean report with a `ProgrammingError` at exit. Reading
    the cache back here is the assertion, since only its owning thread can.
    """
    board = Board(workspace)
    offline = Survey(
        board,
        facts=lambda: HostFacts(hostname="box", memory_total_bytes=10**9),
        reach=lambda alias: "asleep",
        providers=[],
    )
    doctor = Doctor(board, survey=offline, probe=answering(0, _SETTLED))
    assert [found.section for found in doctor.sections()][:4] == [
        "manifest",
        "environment",
        "snapshot",
        "fleet",
    ]
    assert board.dispatcher.cache.hosts() == []


@pytest.mark.parametrize(
    ("found", "verdict", "fix"),
    [
        (
            Snapshot(
                installed=True,
                stale=True,
                detail="the source moved",
                fix=("reinstall", "it"),
            ),
            Verdict.FAIL,
            "reinstall it",
        ),
        (Snapshot(installed=False, detail="running from source"), Verdict.PASS, ""),
    ],
    ids=["a stale snapshot fails with its reinstall", "a checkout passes with its word"],
)
def test_the_snapshot_section_carries_the_staleness_check_into_the_exit_status(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    found: Snapshot,
    verdict: Verdict,
    fix: str,
) -> None:
    """The doctor row is the same check the CLI warning runs, with an exit status behind it."""
    monkeypatch.setattr("mainboard.doctor.staleness.check", lambda: found)
    section = Doctor(Board(workspace)).snapshot()
    assert (section.verdict, section.detail, section.fix) == (verdict, found.detail, fix)


def test_the_runner_bounds_the_probe_it_stages(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate is reached through this workspace's own staged line, under its own deadline."""
    seen: list[tuple[tuple[str, ...], str, float]] = []
    monkeypatch.setattr(
        Provisioner,
        "capture",
        lambda self, command, env, *, timeout: (
            seen.append((tuple(command), env, timeout)) or CommandResult(0, "settled\n", "")
        ),
    )
    status, output = Doctor(Board(workspace)).through_runner("echo settled", 30.0)
    assert (status, output) == (0, "settled\n")
    assert seen == [(("echo", "settled"), "default", 30.0)]
