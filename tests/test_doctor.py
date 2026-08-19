import json
from typing import TYPE_CHECKING

import pytest
from plumbum.commands.processes import ProcessTimedOut

from mainboard import Board, ComputePath, Survey
from mainboard.compute import Access
from mainboard.doctor import Doctor, Section, Verdict
from mainboard.engines.compile.provisioner import Provisioner
from mainboard.engines.compile.state import SyncState

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

_ATPX = "atpx.toml"
_FINGERPRINT = ".pixi-environment-fingerprint"

# A workbench certificate as the tool stamps one, cut down to the two fields mainboard reads.
_BROKEN = json.dumps(
    {"result": {"breakages": [".: failing_claims", "research/x: stale_claims"]}, "exit_status": 1}
)
_SETTLED = json.dumps({"result": {"breakages": []}, "exit_status": 0})


class FixedSurvey(Survey):
    """A survey answering with a fixed set of rows instead of reaching for a machine."""

    def __init__(self, board: Board, rows: Sequence[ComputePath]) -> None:
        super().__init__(board, providers=[])
        self.rows = list(rows)

    def paths(self) -> list[ComputePath]:
        return self.rows


@pytest.fixture
def compiled(workspace: Path) -> Provisioner:
    """A generated workspace whose lock nothing on disk vouches for."""
    provisioner = Provisioner(workspace, Board(workspace).manifest)
    provisioner.out.mkdir(parents=True, exist_ok=True)
    provisioner.pixi.manifest.write_text('[workspace]\nname = "lab"\n', encoding="utf-8")
    provisioner.pixi.lock.write_text("version: 6\n", encoding="utf-8")
    SyncState.path(provisioner.out).write_text(SyncState().render(), encoding="utf-8")
    return provisioner


@pytest.fixture
def provisioned(compiled: Provisioner) -> Callable[[str], None]:
    """Stamp an environment the way a finished pixi installation does."""

    def install(env: str) -> None:
        site = compiled.pixi.env_prefix(env) / "lib" / "python3.14" / "site-packages"
        site.mkdir(parents=True, exist_ok=True)
        fingerprint = compiled.pixi.env_prefix(env) / "conda-meta" / _FINGERPRINT
        fingerprint.parent.mkdir(parents=True, exist_ok=True)
        fingerprint.write_text("installed\n", encoding="utf-8")

    return install


@pytest.fixture
def blessed(compiled: Provisioner, provisioned: Callable[[str], None]) -> Callable[[str], None]:
    """Provision an environment and record the digests a current compile and solve would."""

    def install(env: str) -> None:
        provisioned(env)
        state = SyncState.load(compiled.out)
        current = state.model_copy(
            update={
                "envs": {**state.envs, env: compiled.compiler.digest()},
                "solved_from": compiled.compiler.resolution_digest(),
            }
        )
        SyncState.path(compiled.out).write_text(current.render(), encoding="utf-8")

    return install


def section(sections: Sequence[Section], name: str) -> Section:
    """The one section named `name`."""
    return next(found for found in sections if found.section == name)


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


def test_an_uncompiled_workspace_is_a_word_rather_than_a_failure(workspace: Path) -> None:
    """Nothing was ever built here, which is a first step and not a broken state."""
    found = Doctor(Board(workspace)).environment()
    assert found.verdict is Verdict.WARN
    assert found.detail == "nothing compiled yet"
    assert found.fix == "mainboard install --resolve"


def test_a_lock_solved_from_another_manifest_fails(workspace: Path, compiled: Provisioner) -> None:
    """The lock is what a host installs from, so one nothing vouches for is a broken workspace."""
    found = Doctor(Board(workspace)).environment()
    assert found.verdict is Verdict.FAIL
    assert "pixi.lock was not solved from this manifest" in found.detail
    assert found.fix == "mainboard install --resolve"


def test_an_environment_provisioned_before_the_current_manifest_fails(
    workspace: Path, provisioned: Callable[[str], None]
) -> None:
    """What is installed is not what the manifest describes, and the fix names that env."""
    provisioned("default")
    found = Doctor(Board(workspace)).environment()
    assert "compiled before the current manifest: default" in found.detail
    assert found.fix == "mainboard install default --resolve"


def test_a_fresh_workspace_missing_an_environment_says_which(
    workspace: Path, blessed: Callable[[str], None]
) -> None:
    """Installing what you need is normal, so the unbuilt ones are a word and a next command."""
    blessed("default")
    found = Doctor(Board(workspace)).environment()
    assert found.verdict is Verdict.WARN
    assert "never installed: serving" in found.detail
    assert found.fix == "mainboard install serving"


def test_a_wholly_fresh_workspace_passes(workspace: Path, blessed: Callable[[str], None]) -> None:
    """Every declared environment provisioned, current and importable is the clean state."""
    blessed("default")
    blessed("serving")
    found = Doctor(Board(workspace)).environment()
    assert found.verdict is Verdict.PASS
    assert found.detail == "2 environments provisioned, fresh and whole"


def test_a_wheel_that_lost_its_import_roots_fails(
    workspace: Path, compiled: Provisioner, blessed: Callable[[str], None]
) -> None:
    """No lock notices this, which is exactly why the audit is asked rather than the lock."""
    blessed("default")
    blessed("serving")
    site = compiled.pixi.env_prefix("default") / "lib" / "python3.14" / "site-packages"
    metadata = site / "ghost-1.0.dist-info"
    metadata.mkdir(parents=True)
    metadata.joinpath("METADATA").write_text("Name: ghost\nVersion: 1.0\n")
    metadata.joinpath("INSTALLER").write_text("uv-pixi")
    metadata.joinpath("top_level.txt").write_text("ghost\n")
    found = Doctor(Board(workspace)).environment()
    assert found.verdict is Verdict.FAIL
    assert "needs reinstalling: ghost" in found.detail


def test_a_fleet_with_nothing_to_say_passes(workspace: Path) -> None:
    """Counting what is usable is the whole report when nothing is in the way."""
    board = Board(workspace)
    here = ComputePath(name="local", kind="local", access=Access.HERE, detail="1x 4090")
    found = Doctor(board, survey=FixedSurvey(board, [here])).fleet()
    assert found.verdict is Verdict.PASS
    assert found.detail == "1 paths usable now"
    assert found.fix == ""


def test_a_host_that_is_asleep_never_fails_the_workspace(workspace: Path) -> None:
    """A sleeping host is a fact about the network, not about this code."""
    board = Board(workspace)
    rows = [
        ComputePath(name="local", kind="local", access=Access.HERE, detail="1x 4090"),
        ComputePath(name="gold", kind="ssh", access=Access.REACHABLE, detail="never set up"),
        ComputePath(name="miyabi-g", kind="pbs", access=Access.UNREACHABLE, detail="timed out"),
        ComputePath(name="vast", kind="provider", access=Access.UNKEYED, detail="set VAST_KEY"),
    ]
    found = Doctor(board, survey=FixedSurvey(board, rows)).fleet()
    assert found.verdict is Verdict.WARN
    assert "answering but never set up: gold" in found.detail
    assert "not answering: miyabi-g" in found.detail
    assert "no credentials here: vast" in found.detail
    assert found.fix == "mainboard setup gold"


def test_a_fleet_only_missing_credentials_points_at_the_survey(workspace: Path) -> None:
    """Nothing to set up, so the command showing the whole picture is the useful one."""
    board = Board(workspace)
    row = ComputePath(name="vast", kind="provider", access=Access.UNKEYED, detail="set VAST_KEY")
    assert Doctor(board, survey=FixedSurvey(board, [row])).fleet().fix == "mainboard compute"


def test_a_workbench_reporting_breakages_fails_with_them_named(workspace: Path) -> None:
    """The findings are the workbench's own judgment and are passed on as written."""
    found = Doctor(Board(workspace), math=lambda command: (1, _BROKEN)).math()
    assert found.verdict is Verdict.FAIL
    assert found.detail == "2 breakages: .: failing_claims, research/x: stale_claims"
    assert found.fix == "mainboard run -- atpx doctor"


def test_a_workbench_with_nothing_to_report_passes(workspace: Path) -> None:
    """A clean exit is the workbench saying every claim it holds is settled."""
    found = Doctor(Board(workspace), math=lambda command: (0, _SETTLED)).math()
    assert found.verdict is Verdict.PASS


def test_a_workbench_that_never_ran_is_a_word_not_a_failure(workspace: Path) -> None:
    """A tool nobody installed cannot say the mathematics is broken."""
    probe = Doctor(Board(workspace), math=lambda command: (127, "atpx: command not found\n"))
    found = probe.math()
    assert found.verdict is Verdict.WARN
    assert "exited 127 without a report" in found.detail
    assert found.fix == "mainboard add atpx -l python"


def test_a_workbench_that_will_not_answer_in_time_is_a_word(workspace: Path) -> None:
    """The probe is bounded, and hitting that bound is reported rather than waited out."""

    def hang(command: str) -> tuple[int, str]:
        raise ProcessTimedOut("expired", ["atpx"])

    found = Doctor(Board(workspace), math=hang).math()
    assert found.verdict is Verdict.WARN
    assert "did not answer within 90s" in found.detail


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, False),
        ("[models]\nprover = 'x'\n", False),
        ("[workspace]\nblueprints = 'm'\n", True),
    ],
)
def test_a_math_section_appears_only_for_a_workspace_that_roots_one(
    workspace: Path, config: str | None, expected: bool
) -> None:
    """A config naming no workspace roots no blueprints, so there is nothing to report on."""
    if config is not None:
        (workspace / _ATPX).write_text(config)
    assert Doctor(Board(workspace)).mathematical() is expected


def test_the_sections_cover_the_workspace_and_its_mathematics(workspace: Path) -> None:
    """The four questions asked before starting work, in the order they are asked."""
    (workspace / _ATPX).write_text("[workspace]\nblueprints = 'math'\n")
    board = Board(workspace)
    doctor = Doctor(board, survey=FixedSurvey(board, []), math=lambda command: (0, _SETTLED))
    sections = doctor.sections()
    assert [found.section for found in sections] == ["manifest", "environment", "fleet", "math"]
    assert section(sections, "math").verdict is Verdict.PASS


def test_a_workspace_rooting_no_blueprints_reports_only_the_three(workspace: Path) -> None:
    """A section about mathematics nobody keeps here would be a line that says nothing."""
    board = Board(workspace)
    sections = Doctor(board, survey=FixedSurvey(board, [])).sections()
    assert [found.section for found in sections] == ["manifest", "environment", "fleet"]


def test_the_runner_bounds_the_probe_it_stages(workspace: Path) -> None:
    """The workbench is reached through this workspace's own staged line, under a deadline."""
    status, output = Doctor(Board(workspace)).through_runner("echo settled")
    assert status == 0
    assert "settled" in output


@pytest.mark.parametrize(
    "output", ["", "atpx: not found", "{not json", '{"result": {"breakages": 3}}']
)
def test_output_carrying_no_readable_certificate_reports_no_breakages(
    workspace: Path, output: str
) -> None:
    """Failing without a report and failing with one are different findings."""
    found = Doctor(Board(workspace), math=lambda command: (1, output)).math()
    assert found.verdict is Verdict.WARN
