from pathlib import Path

from mainboard.dispatch.provenance import Source
from mainboard.dispatch.shared import (
    CLOSURE_VAR,
    COMMIT_VAR,
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
    closure = Closure.of(target, root=lab.root, distributions=Lab.DISTRIBUTIONS)
    shipment = Shipment.of_closure(closure, root=lab.root)
    assert shipment.sealed
    assert shipment.command == f"python -m {runner()} {Lab.JOB}::app -- --x 3"
    assert shipment.spelling == f"{Lab.JOB}::app --x 3"
    assert shipment.imports == closure.roots
    assert shipment.needs == ("data/corpus",)
    assert shipment.fetch == "research/camp/experiments/node/evidence"
    assert shipment.first_party == ("core", "experiments", "sub")
    assert shipment.listing.splitlines()[0].startswith("mainboard.toml\t")
    assert len(shipment.listing.splitlines()) == len(closure.files)
    assert shipment.listing_name == f"closure-{shipment.source.digest[:12]}.tsv"
    exported = shipment.exports("/pinned/closure.tsv")
    assert exported[CLOSURE_VAR] == "/pinned/closure.tsv"
    assert exported[FIRST_PARTY_VAR] == "core:experiments:sub"
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
