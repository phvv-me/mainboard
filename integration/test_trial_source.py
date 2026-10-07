"""Local staged listings and remote snapshot markers verify the same source boundary."""

from pathlib import Path

import pytest

from mainboard.core.project import Project
from mainboard.dispatch.dispatcher import Dispatcher
from mainboard.dispatch.provenance import SourceTree, listing
from mainboard.dispatch.shared import CLOSURE_VAR, DIGEST_VAR
from mainboard.dispatch.shipment import Shipment
from mainboard.jobs.closure import Closure
from mainboard.jobs.target import Target
from mainboard.trials import provenance
from mainboard.trials.provenance import Admissibility, Card, Preflight, source


@pytest.mark.parametrize("placement", ["local", "snapshot"])
def test_captured_trials_verify_both_listing_locations(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, placement: str
) -> None:
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
    if placement == "local":
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

    taken = Preflight(lane.parent, lane.parent)
    assert taken.source.root == workspace.resolve()
    assert taken.digest == shipment.source.digest
    assert taken.source.mirrored
    assert taken.admits(lane) is Admissibility.ADMISSIBLE
    unlisted = lane.with_name("test_unlisted.py")
    unlisted.write_bytes(lane.read_bytes())
    assert taken.admits(unlisted) is Admissibility.UNRECORDED
    with pytest.raises(RuntimeError, match="outside the captured source workspace"):
        source(workspace.parent)

    with monkeypatch.context() as changed:
        for name in DIGEST_VAR.names:
            changed.setenv(name, "0" * 64)
        with pytest.raises(RuntimeError, match="listing does not match"):
            source(lane.parent)
    closure = Path(CLOSURE_VAR.read())
    closure.write_text(shipment.listing + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(RuntimeError, match="listing does not match"):
        source(lane.parent)
    closure.write_text(shipment.listing, encoding="utf-8", newline="\n")
    lane.write_bytes(b"def test_lane():\n    assert False\n")
    assert taken.admits(lane) is Admissibility.UNRECORDED
    with pytest.raises(RuntimeError, match="source bytes differ"):
        source(lane.parent)


def test_snapshot_boundary_does_not_expand_to_a_composing_workspace(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "mb.toml").write_text(
        '[workspace]\nname = "outer"\nmembers = ["snapshot"]\n',
        encoding="utf-8",
        newline="\n",
    )
    snapshot = workspace / "snapshot"
    snapshot.mkdir()
    (snapshot / "mb.toml").write_bytes(b'[workspace]\nname = "captured"\n')
    captured, rows = SourceTree(snapshot).seal(["mb.toml"])
    closure = snapshot / Project().marker("closure")
    closure.write_text(listing(rows), encoding="utf-8", newline="\n")
    for name, value in {
        **CLOSURE_VAR.exported(str(closure)),
        **DIGEST_VAR.exported(captured.digest),
    }.items():
        monkeypatch.setenv(name, value)
    assert Project().find_root(snapshot) == workspace
    assert source(snapshot).root == snapshot.resolve()
    with pytest.raises(RuntimeError, match="outside the captured source workspace"):
        source(workspace)
