import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from mainboard.cli import build
from mainboard.dispatch import HostSetup
from mainboard.dispatch.state import Cache, RunRecord

if TYPE_CHECKING:
    from pathlib import Path

    from .support import Relayed

_FIELD_VALUE_HEADER = "field\tvalue"

# One run as the jobs table projects it, the row every case below is a variation on.
_ROW = {
    "state": "ok",
    "host": "gold",
    "name": "train",
    "handle": "H1",
    "submitted_at": "2026-08-01T00:00:00",
}


def seed_run(handle: str = "H1", submitted_at: str = "2026-08-01T00:00:00") -> None:
    """Record one dispatched run in the shared cache, the way a submit would have."""
    Cache().record(
        RunRecord(
            handle=handle,
            target="gold",
            kind="ssh",
            script="job.sh",
            args="",
            git_sha="abc1234",
            dirty=0,
            submitted_at=submitted_at,
            name="train",
            state="ok",
        )
    )


def onboarded(host: str = "gold") -> HostSetup:
    """What onboarding recorded for a host, the row the hosts table reads back."""
    return HostSetup(
        host=host,
        root="/repo",
        env="default",
        activate="/repo/.mainboard/activate.sh",
        installer="uv",
        rejected=(("pip", "reported unavailable"),),
        tool="0.1.0",
        onboarded_at="2026-08-17T00:00:00+00:00",
    )


@pytest.mark.parametrize(
    ("flags", "fragments"),
    [
        (["--json", "--fields", "hostname,schema_version"], ()),
        ([], ("hostname", "facts")),
        (["--agent"], (_FIELD_VALUE_HEADER, "hostname")),
    ],
    ids=["a projection over the probed fields", "the default rich table", "the compact record"],
)
def test_the_facts_verb_prints_this_machines_own_probe(
    depot: Path, capsys: pytest.CaptureFixture[str], flags: list[str], fragments: tuple[str, ...]
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["facts", *flags])
    out = capsys.readouterr().out
    if not fragments:
        payload = json.loads(out)
        assert set(payload) == {"hostname", "schema_version"}
        assert payload["schema_version"] >= 1
        return
    assert all(fragment in out for fragment in fragments)


@pytest.mark.parametrize(
    ("flags", "fragments"),
    [
        (["--json"], ()),
        ([], ("setup", "gold")),
    ],
    ids=["the record as json", "the default rich table"],
)
def test_the_setup_verb_shows_what_the_host_became(
    depot: Path,
    relayed: Sequence[Relayed],
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    fragments: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["setup", "gold", *flags])
    out = capsys.readouterr().out
    if not fragments:
        assert json.loads(out)["installer"] == "uv"
        return
    assert all(fragment in out for fragment in fragments)


@pytest.mark.parametrize(
    ("seeded", "flags", "expected"),
    [
        (["H1"], [], [_ROW]),
        (["H1"], ["--fields", "handle,state"], [{"handle": "H1", "state": "ok"}]),
        (
            ["H1", "H2"],
            ["--limit", "1"],
            [{**_ROW, "handle": "H2", "submitted_at": "2026-08-02T00:00:00"}],
        ),
        ([], [], []),
    ],
    ids=[
        "every projected field of one run",
        "a projection over two of them",
        "the newest run only, under the limit",
        "a cache nobody has dispatched from yet",
    ],
)
def test_the_jobs_verb_lists_recent_runs_newest_first(
    depot: Path,
    capsys: pytest.CaptureFixture[str],
    seeded: Sequence[str],
    flags: list[str],
    expected: list[dict[str, str]],
) -> None:
    for index, handle in enumerate(seeded):
        seed_run(handle, submitted_at=f"2026-08-0{index + 1}T00:00:00")
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["jobs", "--json", *flags])
    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([], ("H1", "gold", "jobs")),
        (["--agent"], ("state\thost\tname\thandle\tsubmitted_at", "H1")),
    ],
    ids=["the default rich table", "the compact table"],
)
def test_the_jobs_verb_tables_what_it_listed(
    depot: Path, capsys: pytest.CaptureFixture[str], flags: list[str], expected: Sequence[str]
) -> None:
    seed_run()
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["jobs", *flags])
    out = capsys.readouterr().out
    assert all(fragment in out for fragment in expected)


def test_the_hosts_verb_lists_the_recorded_onboardings(
    depot: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Cache().save_host(onboarded())
    with pytest.raises(SystemExit, match="0"):
        build(depot)(["hosts", "--json"])
    [payload] = json.loads(capsys.readouterr().out)
    assert payload["host"] == "gold"
    assert payload["installer"] == "uv"
    assert payload["onboarded_at"]
