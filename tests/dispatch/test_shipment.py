import os
import shlex
from pathlib import Path

import pytest

from mainboard import MissionError
from mainboard.dispatch.provenance import Source
from mainboard.dispatch.shared import (
    CLOSURE_VAR,
    COMMIT_VAR,
    DEFERRED_VAR,
    DIGEST_VAR,
    FIRST_PARTY_VAR,
    SOURCE_VAR,
)
from mainboard.dispatch.shipment import Shipment, runner
from mainboard.jobs.closure import Closure
from mainboard.jobs.target import Target

from ..support import Lab


def test_a_command_ships_the_mirror_under_the_trees_provenance_and_exports_only_what_it_has() -> (
    None
):
    source = Source(identity="v1", key="v1", commit="c" * 40, digest="d" * 64)
    shipment = Shipment.of_command("python -m foo", source=source, imports=("src",))
    assert not shipment.sealed
    assert (shipment.command, shipment.spelling) == ("python -m foo", "python -m foo")
    assert shipment.exports() == {SOURCE_VAR: "v1", COMMIT_VAR: "c" * 40, DIGEST_VAR: "d" * 64}
    bare = Shipment.of_command("true", source=Source(identity="", key="untracked"), imports=())
    assert bare.exports() == {}
    assert bare.locally(Path("/w")) == ["env", "true"]


def test_a_job_ships_its_closure_and_runs_through_the_one_runner(lab: Lab) -> None:
    target = Target.spelled([Lab.JOB, "--x", "3"], lab.root)
    assert target is not None
    closure = Closure.of(
        target,
        root=lab.root,
        distributions=Lab.DISTRIBUTIONS,
        environment=lab.root / Lab.ENVIRONMENT,
    )
    shipment = Shipment.of_closure(closure, root=lab.root)
    shipment.admit(lab.root)
    assert shipment.sealed
    assert shipment.command == f"python -m {runner()} {Lab.JOB}::app -- --x 3"
    assert shipment.spelling == f"{Lab.JOB}::app --x 3"
    assert shipment.imports == closure.roots
    assert shipment.needs == ("data/corpus",)
    assert shipment.fetch == "research/camp/experiments/node/evidence"
    assert shipment.first_party == ("core", "experiments", "sub")
    assert shipment.deferred == ()
    assert shipment.listing.splitlines()[0].startswith("mainboard.toml\t")
    assert len(shipment.listing.splitlines()) == len(closure.files)
    assert shipment.listing_name == f"closure-{shipment.source.digest[:12]}.tsv"
    exported = shipment.exports("/pinned/closure.tsv")
    assert exported[CLOSURE_VAR] == "/pinned/closure.tsv"
    assert exported[FIRST_PARTY_VAR] == "core:experiments:sub"
    assert DEFERRED_VAR not in exported
    assert exported[SOURCE_VAR] == shipment.source.identity
    local = shipment.locally(lab.root, closure=".mainboard/dispatch/jobs/closure.tsv")
    assert local[:2] == [
        "env",
        "PYTHONPATH=" + ":".join(str(lab.root / place) for place in closure.roots),
    ]
    assert f"{CLOSURE_VAR}={lab.root}/.mainboard/dispatch/jobs/closure.tsv" in local
    assert local[-6:] == ["python", "-m", runner(), f"{Lab.JOB}::app", "--", "--x"] or local[
        -7:-1
    ] == ["python", "-m", runner(), f"{Lab.JOB}::app", "--", "--x"]


def test_local_import_roots_use_the_native_path_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    shipment = Shipment.of_command(
        "python -m job", source=Source(identity="", key="untracked"), imports=("a", "b")
    )
    monkeypatch.setattr(os, "pathsep", ";")
    assert shipment.local_exports(Path("/workspace"))["PYTHONPATH"] == "/workspace/a;/workspace/b"
    assert "PYTHONPATH=/workspace/a:/workspace/b" in shipment.locally(Path("/workspace"))


def test_a_deferred_distribution_rides_the_shipment_and_exports_for_the_runner(
    lab: Lab,
) -> None:
    """What `Closure.deferred` finds ships nothing, and rides the shipment as its own variable."""
    lab.write("packages/ext/src/ext/__init__.py", "")
    lab.write(
        "research/camp/experiments/node/run.py", "import ext\n\n\ndef main() -> None:\n    pass\n"
    )
    environment = lab.compiled(
        "camp-ext",
        lab.root / f"{Lab.ENVIRONMENT}/lib/python3.14/site-packages/ext/"
        "_native.cpython-314-x86_64-linux-gnu.so",
    )
    target = Target.spelled([Lab.JOB], lab.root)
    assert target is not None
    closure = Closure.of(
        target,
        root=lab.root,
        distributions=(*Lab.DISTRIBUTIONS, "packages/ext/src"),
        environment=environment,
    )
    shipment = Shipment.of_closure(closure, root=lab.root)
    assert shipment.deferred == ("ext",)
    assert shipment.exports()[DEFERRED_VAR] == "ext"
    assert not any(line.startswith("packages/ext/src/") for line in shipment.listing.splitlines())


@pytest.mark.parametrize("changed", ["node.md", "run.py", "packages/sub/src/sub/thing.py"])
@pytest.mark.parametrize("late", [False, True])
def test_research_admission_refuses_dirty_or_stale_source(
    lab: Lab, changed: str, late: bool
) -> None:
    target = Target.spelled([Lab.JOB], lab.root)
    assert target is not None
    path = changed if changed.startswith("packages/") else f"{target.node}/{changed}"
    closure = Closure.of(
        target,
        root=lab.root,
        distributions=Lab.DISTRIBUTIONS,
        environment=lab.root / Lab.ENVIRONMENT,
    )
    before = Shipment.of_closure(closure, root=lab.root)
    lab.write(path, (lab.root / path).read_text() + "\n# changed\n")
    shipment = before if late else Shipment.of_closure(closure, root=lab.root)
    with pytest.raises(MissionError, match="changed after|clean committed"):
        shipment.admit(lab.root)


@pytest.mark.parametrize("removed", ["node.md", "run.py"])
def test_research_admission_refuses_disappearing_files(lab: Lab, removed: str) -> None:
    target = Target.spelled([Lab.JOB], lab.root)
    assert target is not None
    closure = Closure.of(
        target,
        root=lab.root,
        distributions=Lab.DISTRIBUTIONS,
        environment=lab.root / Lab.ENVIRONMENT,
    )
    shipment = Shipment.of_closure(closure, root=lab.root)
    (lab.root / target.node / removed).unlink()
    with pytest.raises((MissionError, FileNotFoundError)):
        shipment.admit(lab.root)


def test_research_admission_keeps_historical_seals_and_never_imports_the_job(lab: Lab) -> None:
    target = Target.spelled([Lab.JOB], lab.root)
    assert target is not None
    node = lab.write(
        f"{target.node}/node.md",
        "---\nstatus: refuted\nregistration_sha256: preserved-retired-seal\n---\n",
    )
    lab.write(Lab.JOB, "raise RuntimeError('must not import')\napp = None\n")
    lab.commit()
    closure = Closure.of(
        target,
        root=lab.root,
        distributions=Lab.DISTRIBUTIONS,
        environment=lab.root / Lab.ENVIRONMENT,
    )
    original = node.read_bytes()
    shipment = Shipment.of_closure(closure, root=lab.root)
    shipment.admit(lab.root)
    assert node.read_bytes() == original
    wrong = shipment.model_copy(
        update={"source": shipment.source.model_copy(update={"digest": "wrong"})}
    )
    with pytest.raises(MissionError, match="listing does not match"):
        wrong.admit(lab.root)


def test_ordinary_dirty_software_and_commands_keep_their_existing_admission(lab: Lab) -> None:
    path = "packages/tool/tests/experiments/test_case.py"
    lab.write(path, "def test_case():\n    raise RuntimeError('must not import')\n")
    target = Target.spelled([path], lab.root)
    assert target is not None and not target.registration
    closure = Closure.of(
        target, root=lab.root, distributions=(), environment=lab.root / Lab.ENVIRONMENT
    )
    shipment = Shipment.of_closure(closure, root=lab.root)
    assert shipment.source.dirty
    shipment.admit(lab.root)
    Shipment.of_command("python -m research.work", source=shipment.source, imports=()).admit(
        lab.root
    )


@pytest.mark.parametrize("name", ["test_law", "test_law[a b]", "test_law[a'b]"])
@pytest.mark.parametrize("prefix", ["", "packages/../"])
def test_real_closures_quote_paths_and_parameter_ids_before_admission(
    lab: Lab, name: str, prefix: str
) -> None:
    file = "research/project with spaces/experiments/node/test_law.py"
    lab.write(file, "def test_law():\n    raise RuntimeError('must not import')\n")
    node = lab.write("research/project with spaces/experiments/node/node.md", "# registered\n")
    lab.commit()
    target = Target.spelled([f"{prefix}{file}::{name}"], lab.root)
    assert target is not None
    closure = Closure.of(
        target, root=lab.root, distributions=(), environment=lab.root / Lab.ENVIRONMENT
    )
    shipment = Shipment.of_closure(closure, root=lab.root)
    shipment.admit(lab.root)
    node.write_text("changed after sealing\n")
    with pytest.raises(MissionError, match="changed after Mainboard prepared"):
        shipment.admit(lab.root)
    assert shlex.split(shipment.spelling) == [f"{file}::{name}"]
