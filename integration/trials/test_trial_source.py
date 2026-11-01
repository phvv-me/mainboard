"""Local staged listings and remote snapshot markers verify the same source boundary."""

from pathlib import Path

import pytest
from patos import FrozenModel, Runtime

from mainboard import Project
from mainboard.dispatch import Dispatcher, Shipment, SourceTree
from mainboard.dispatch.provenance import listing
from mainboard.dispatch.shared import CLOSURE_VAR, DIGEST_VAR
from mainboard.jobs import Closure, Target
from mainboard.trials import Admissibility, Card, Preflight, provenance, source


class Captured(FrozenModel):
    """A lane inside a captured source tree and the preflight taken of it.

    lane: the lane file the shipment lists.
    shipment: what was captured, whose listing the environment points at.
    taken: the preflight taken before the lane or the listing changes.
    """

    lane: Path
    shipment: Shipment
    taken: Runtime[Preflight]


@pytest.fixture(params=["local", "snapshot"])
def captured(
    request: pytest.FixtureRequest, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> Captured:
    """A lane captured with its listing staged locally or marked as a remote snapshot."""
    lane = workspace / "research" / "lab" / "experiments" / "law" / "test_lane.py"
    lane.parent.mkdir(parents=True)
    lane.write_bytes(b"def test_lane():\n    assert True\n")
    relative = lane.relative_to(workspace).as_posix()
    shipment = Shipment.of_closure(
        Closure(
            target=Target(file=relative, name=""),
            files=("mb.toml", relative),
            roots=("research/lab",),
        ),
        root=workspace,
    )
    if request.param == "local":
        staged = Dispatcher(root=workspace).stage_listing(shipment)
        exported = shipment.local_exports(workspace, closure=staged)
    else:
        closure = workspace / Project().marker("closure")
        closure.write_text(shipment.listing, encoding="utf-8", newline="\n")
        exported = shipment.exports(str(closure))
    for name, value in exported.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(provenance, "card_of", lambda machine: Card())
    monkeypatch.chdir(workspace.parent)
    return Captured(lane=lane, shipment=shipment, taken=Preflight(lane.parent, repo=lane.parent))


def test_a_captured_lane_is_admitted_and_an_unlisted_sibling_is_not(
    workspace: Path, captured: Captured
) -> None:
    taken, lane = captured.taken, captured.lane
    assert taken.source.root == workspace.resolve()
    assert taken.digest == captured.shipment.source.digest
    assert taken.source.mirrored
    assert taken.admits(lane) is Admissibility.ADMISSIBLE
    unlisted = lane.with_name("test_unlisted.py")
    unlisted.write_bytes(lane.read_bytes())
    assert taken.admits(unlisted) is Admissibility.UNRECORDED


@pytest.mark.usefixtures("captured")
def test_a_source_outside_the_captured_workspace_is_refused(workspace: Path) -> None:
    with pytest.raises(RuntimeError, match="outside the captured source workspace"):
        source(workspace.parent)


def test_a_listing_checked_against_a_wrong_digest_is_refused(
    captured: Captured, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in DIGEST_VAR.names:
        monkeypatch.setenv(name, "0" * 64)
    with pytest.raises(RuntimeError, match="listing does not match"):
        source(captured.lane.parent)


def test_a_listing_with_a_changed_text_is_refused_by_its_digest(captured: Captured) -> None:
    closure = Path(CLOSURE_VAR.read())
    closure.write_text(captured.shipment.listing + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(RuntimeError, match="listing does not match"):
        source(captured.lane.parent)


def test_a_lane_whose_bytes_changed_is_unrecorded_and_refused(captured: Captured) -> None:
    captured.lane.write_bytes(b"def test_lane():\n    assert False\n")
    assert captured.taken.admits(captured.lane) is Admissibility.UNRECORDED
    with pytest.raises(RuntimeError, match="source bytes differ"):
        source(captured.lane.parent)


@pytest.fixture
def snapshot(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A captured snapshot folder, which the workspace around it names as a member."""
    (workspace / "mb.toml").write_text(
        '[workspace]\nname = "outer"\nmembers = ["snapshot"]\n',
        encoding="utf-8",
        newline="\n",
    )
    root = workspace / "snapshot"
    root.mkdir()
    (root / "mb.toml").write_bytes(b'[workspace]\nname = "captured"\n')
    sealed, rows = SourceTree(root).seal(["mb.toml"])
    closure = root / Project().marker("closure")
    closure.write_text(listing(rows), encoding="utf-8", newline="\n")
    for name, value in {
        **CLOSURE_VAR.exported(str(closure)),
        **DIGEST_VAR.exported(sealed.digest),
    }.items():
        monkeypatch.setenv(name, value)
    return root


def test_snapshot_boundary_does_not_expand_to_a_composing_workspace(
    workspace: Path, snapshot: Path
) -> None:
    assert Project().find_root(snapshot) == workspace
    assert source(snapshot).root == snapshot.resolve()
    with pytest.raises(RuntimeError, match="outside the captured source workspace"):
        source(workspace)
