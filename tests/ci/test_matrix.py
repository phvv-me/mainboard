import threading
from pathlib import Path

import pytest

from mainboard import Board, MissionError
from mainboard.ci import Leg, LocalLeg, Matrix, Mirror, Package, RemoteLeg, Result, Verdict
from mainboard.ci.definition import Family, Step
from mainboard.context.plan import ExecutionPlan

from .conftest import declare, plan, say, step

_HOSTS = """
[hosts.win]
kind = "ssh"
platform = "win-64"
root = "C:/m"

[hosts.lin]
kind = "ssh"
platform = "linux-64"
root = "/home/m"

[hosts.mac]
kind = "ssh"
platform = "osx-arm64"
root = "/Users/m"
"""


def _workspace(root: Path, hosts: list[str]) -> Board:
    """A workspace declaring one host per platform, `hosts` of them named for the matrix."""
    root.mkdir(parents=True, exist_ok=True)
    listed = ", ".join(f'"{host}"' for host in hosts)
    (root / "mainboard.toml").write_text(
        f'[workspace]\nname = "w"\n{_HOSTS}\n[ci]\nhosts = [{listed}]\n', encoding="utf-8"
    )
    return Board(root)


@pytest.mark.parametrize("supported", [["linux", "osx", "win"], ["linux"]], ids=["all", "linux"])
def test_the_matrix_is_this_machine_plus_one_host_of_every_other_supported_family(
    tmp_path: Path, supported: list[Family]
) -> None:
    """A host of this machine's own family is passed over; an unsupported one is never a leg."""
    board = _workspace(tmp_path / "w", ["win", "lin", "mac"])
    os_line = f"os = {supported!r}\n".replace("'", '"')
    package = Package.found(declare(tmp_path / "w" / "packages" / "p", os_line + step("s", "t")))

    matrix = Matrix.planned(package, board)

    here = LocalLeg(package.root).family
    remote = [
        family for family in ("win", "linux", "osx") if family != here and family in supported
    ]
    local = [here] if here in supported else []
    assert [leg.family for leg in matrix.legs] == [*local, *remote]
    assert isinstance(matrix.legs[0], LocalLeg) is bool(local)
    assert all(leg.package == "packages/p" for leg in matrix.legs if isinstance(leg, RemoteLeg))
    assert matrix.uncovered == []


def test_a_family_no_leg_runs_on_is_named(tmp_path: Path) -> None:
    board = _workspace(tmp_path / "w", [])
    package = Package.found(declare(tmp_path / "w" / "p", step("s", "t")))
    matrix = Matrix.planned(package, board)
    here = LocalLeg(package.root).family
    assert matrix.uncovered == [family for family in ("linux", "osx", "win") if family != here]


def test_a_package_outside_the_workspace_has_no_host_to_ship_to(tmp_path: Path) -> None:
    board = _workspace(tmp_path / "w", ["lin"])
    package = Package.found(declare(tmp_path / "elsewhere", step("s", "t")))
    with pytest.raises(MissionError, match="outside the workspace"):
        Matrix.planned(package, board)


class _Paced(Leg):
    """A leg whose steps wait on a barrier every leg reaches, so only a parallel run finishes."""

    def __init__(self, name: str, family: Family, barrier: threading.Barrier) -> None:
        super().__init__(name, family)
        self.barrier = barrier

    def _step(self, step: Step) -> Result:
        self.barrier.wait(timeout=10)
        return Result(leg=self.name, os=self.family, step=step.name, verdict=Verdict.OK)


def test_every_leg_runs_at_once_and_reports_in_declared_order(tmp_path: Path) -> None:
    package = Package.found(declare(tmp_path, step("one", say("1")) + step("two", say("2"))))
    barrier = threading.Barrier(3)
    legs = [
        _Paced(name, family, barrier)
        for name, family in [("a", "linux"), ("b", "osx"), ("c", "win")]
    ]

    results = Matrix(package, legs).run()

    assert [(result.leg, result.step) for result in results] == [
        (leg, name) for leg in "abc" for name in ("one", "two")
    ]


class _Dispatcher:
    """Records what a mirror asks the dispatch core to send."""

    def __init__(self) -> None:
        self.sent: list[tuple[ExecutionPlan, str]] = []

    def rsync_up(self, target: ExecutionPlan, root: str) -> list[str]:
        self.sent.append((target, root))
        return list(target.profile.sync.include)


def test_a_mirror_sends_the_package_alone_and_keeps_the_hosts_own_rules() -> None:
    dispatcher = _Dispatcher()
    original = plan("lin", "linux-64")

    Mirror(dispatcher)(original, "/m/.mainboard/ci", "packages/p")

    ((sent, root),) = dispatcher.sent
    assert root == "/m/.mainboard/ci"
    assert sent.profile.sync.include == ["packages/p"]
    assert sent.profile.sync.exclude == ["data/raw"]
    assert sent.profile.sync.protect == ["results/***"]
    assert original.profile.sync.include == ["everything"]
