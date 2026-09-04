from pathlib import Path

import pytest

from mainboard.dispatch import snapshots as snapshots_module
from mainboard.dispatch.snapshots import (
    STAMP,
    HostUnreachable,
    Snapshots,
    containers,
    source_key,
    writable,
)

from .support import machine_with


@pytest.fixture
def committed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer every local git question with nothing, the way a clean tree's delta reads."""
    monkeypatch.setattr(snapshots_module, "git", lambda *args: "")


def test_a_committed_tree_is_named_by_the_identity_its_receipts_already_carry(
    committed: None, tmp_path: Path
) -> None:
    """Two dispatches of one commit must land on one snapshot, or the reuse buys nothing."""
    del committed
    key = source_key(tmp_path, source="v0.4.8-3-gb62c31e")
    assert key == "v0.4.8-3-gb62c31e"
    assert source_key(tmp_path, source="v0.4.8-3-gb62c31e") == key


def test_a_key_can_never_name_a_path_outside_the_sources_directory(
    committed: None, tmp_path: Path
) -> None:
    """A source identity is git's text, and text reaches the shell that builds the tree."""
    del committed
    assert source_key(tmp_path, source="../../etc/passwd") == "-..-etc-passwd"
    assert source_key(tmp_path, source="..") == "untracked"


def test_a_dirty_tree_keys_on_its_delta_so_a_later_edit_is_never_handed_the_earlier_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`-dirty` names no commit, so the working tree itself has to separate two dispatches."""
    delta = {"value": "M a.py"}
    monkeypatch.setattr(snapshots_module, "git", lambda *args: delta["value"])
    first = source_key(tmp_path, source="abc1234-dirty")
    assert first.startswith("abc1234-dirty-")
    assert source_key(tmp_path, source="abc1234-dirty") == first
    delta["value"] = "M a.py\nM b.py"
    assert source_key(tmp_path, source="abc1234-dirty") != first


def test_a_workspace_with_no_git_at_all_still_gets_a_key(committed: None, tmp_path: Path) -> None:
    """An empty identity is what a mirror without history reads, and it still has to dispatch."""
    del committed
    assert source_key(tmp_path, source="").startswith("untracked-")


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        (["scripts", "mainboard.toml"], ["."]),
        (["research/compression", "research/bale"], [".", "research"]),
        (["a/b/c"], [".", "a", "a/b"]),
    ],
)
def test_every_directory_a_snapshot_creates_is_one_it_has_to_fill_from_the_mirror(
    sources: list[str], expected: list[str]
) -> None:
    """A directory the copy invented holds none of the data dirs the mirror keeps beside it."""
    assert containers(sources) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("out/raw", "out/raw"),
        ("", ""),
        ("/absolute/raw", ""),
        ("../outside", ""),
        ("out/../../outside", ""),
    ],
)
def test_only_a_workspace_relative_results_path_is_ever_spliced_into_the_shell(
    path: str, expected: str
) -> None:
    """The results path is caller-typed and the snapshot shell removes what it links over."""
    assert writable(path) == expected


def test_pinning_copies_the_shipped_set_by_hardlink_and_links_the_rest_back(
    committed: None,
) -> None:
    """The whole bargain: code frozen, environment and data reached live through the mirror."""
    del committed
    remote = machine_with()
    pinned = Snapshots("/work/projects").pin(
        remote,
        key="abc1234",
        sources=["research/compression", "mainboard.toml"],
        results="research/compression/raw",
        filters=[":- .gitignore"],
        exclude=[".git"],
    )
    assert pinned == "/work/projects/.mainboard/dispatch/sources/abc1234"
    [program] = remote.lines
    assert f"if [ -f {pinned}/{STAMP} ]; then exit 0; fi" in program
    assert "cd /work/projects" in program
    assert "--link-dest=/work/projects/" in program
    assert "rsync -aR" in program
    assert "research/compression mainboard.toml" in program
    # Every container directory the copy created is filled from the mirror, which is what puts
    # `.mainboard/` (the environment, the logs, the staged script) back within the job's reach.
    assert "for d in . research; do" in program
    assert 'ln -s "$e"' in program
    # The declared results path is a symlink back to the mirror, so what the job writes there is
    # what the pull already goes looking for.
    assert (
        f"ln -s /work/projects/research/compression/raw {pinned}/research/compression/raw"
        in program
    )
    assert program.endswith(f"printf '%s\\n' abc1234 > {pinned}/{STAMP}")


def test_pinning_a_tree_the_host_could_not_build_refuses_instead_of_dispatching_into_it(
    committed: None,
) -> None:
    """A job started in a half-built tree imports whatever happened to be copied first."""
    del committed
    remote = machine_with(rules=[("rsync", 23, "rsync: link_stat failed")])
    with pytest.raises(SystemExit, match="could not pin the source tree"):
        Snapshots("/work/projects").pin(remote, key="abc1234", sources=["src"])


def test_a_host_that_dropped_while_pinning_reads_as_unreachable_not_as_a_broken_tree(
    committed: None,
) -> None:
    """A transport blip is a fact about the host, and the dispatch retries rather than refuses."""
    del committed
    remote = machine_with(rules=[("rsync", 255, "ssh: connect to host gold port 22: timed out")])
    with pytest.raises(HostUnreachable):
        Snapshots("/work/projects").pin(remote, key="abc1234", sources=["src"])


def test_pruning_keeps_the_newest_few_and_everything_a_live_job_still_runs_from() -> None:
    """A tree a queued job is pinned to outlives the sweep however old the directory is."""
    remote = machine_with("new\nolder\noldest\nancient\nlive\n")
    dropped = Snapshots("/work/projects", keep=2).prune(remote, live={"live"})
    assert dropped == ["oldest", "ancient"]
    removal = remote.lines[-1]
    assert removal.startswith("rm -rf ")
    assert "/work/projects/.mainboard/dispatch/sources/oldest" in removal
    assert "live" not in removal


def test_pruning_a_host_with_nothing_to_drop_never_runs_a_removal() -> None:
    """A sweep runs every twenty minutes, so the quiet case has to cost one listing and no more."""
    remote = machine_with("only\n")
    assert Snapshots("/work/projects").prune(remote, live=set()) == []
    assert not remote.ran("rm -rf")
