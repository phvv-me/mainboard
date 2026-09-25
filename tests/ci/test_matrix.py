import os
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from mainboard import Board, MissionError
from mainboard.ci import Leg, LocalLeg, Matrix, Package, PackageMirror, RemoteLeg, Result, Verdict
from mainboard.ci.definition import Family, Step
from mainboard.dispatch.agent import Agent
from mainboard.dispatch.dispatcher import Dispatcher

from ..dispatch.support import InProcessLink
from .conftest import declare, plan, say, step

# This interpreter as a shell would have it typed, which a path with a space needs quoted.
_PYTHON = (
    subprocess.list2cmdline([sys.executable]) if os.name == "nt" else shlex.quote(sys.executable)
)

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


def test_a_mirror_sends_the_package_alone_and_keeps_the_hosts_own_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the package lands under the gate's root, less what the host excludes everywhere."""
    board = _workspace(tmp_path / "w", ["lin"])
    for relative in ("packages/p/src/m.py", "packages/p/data/raw/big.bin", "research/r.py"):
        path = board.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    agent = Agent(InProcessLink(), python=_PYTHON)
    monkeypatch.setattr(Dispatcher, "agent", lambda _self, _plan: agent)
    target = tmp_path / "host" / ".mainboard" / "ci"

    PackageMirror(board.dispatcher)(plan("lin", "linux-64"), str(target), "packages/p")

    shipped = sorted(path.relative_to(target).as_posix() for path in target.rglob("*.py"))
    assert shipped == ["packages/p/src/m.py"]
    assert not (target / "packages/p/data/raw").exists()
