"""What `host evict` removes from a host: only files the lake holds intact at the same path."""

import hashlib
import io
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import Field

from mainboard.dispatch import SshTransport
from mainboard.dispatch.collection.collector import Collector, Eviction
from mainboard.dispatch.collection.pack import Center, evict
from mainboard.state import Evidence

_RUN = "results/run"

# What the center's lake holds, `other.bin` with other bytes than the host's.
_CENTER = {
    f"{_RUN}/held.bin": b"held",
    f"{_RUN}/deep/a/b/held.txt": b"deep",
    f"{_RUN}/other.bin": b"center bytes",
    "outside.bin": b"held",
}

# What the host holds: `new.bin` the lake never saw, `held.bin`'s bytes at `elsewhere.bin`.
_HOST = {
    f"{_RUN}/held.bin": b"held",
    f"{_RUN}/deep/a/b/held.txt": b"deep",
    f"{_RUN}/other.bin": b"host bytes",
    f"{_RUN}/new.bin": b"new",
    f"{_RUN}/elsewhere.bin": b"held",
    "outside.bin": b"held",
}

_UNINDEXED = "the lake keeps no intact copy at this path"


class _Host(SshTransport):
    """An ssh transport whose host is a local folder, running each script it gets as a host does.

    The script runs on this Python, isolated, so it imports the standard library alone.
    """

    scripts: list[str] = Field(default_factory=list)

    def command(self, host: str) -> tuple[str, ...]:
        return ("ssh", host)

    def run(
        self,
        command: Sequence[str],
        host: str,
        *,
        operation: str,
        input_text: str | None = None,
        output: Path | None = None,
    ) -> str:
        self.scripts.append(input_text or "")
        with (output or Path(os.devnull)).open("wb") as sink:
            subprocess.run(
                [sys.executable, "-I", "-"],
                input=(input_text or "").encode(),
                stdout=sink,
                check=True,
            )
        return ""


def _left(root: Path) -> set[str]:
    """Every entry under `root` but a folder, links included, by its path relative to `root`."""
    return {
        (Path(folder) / name).relative_to(root).as_posix()
        for folder, _, names in os.walk(root)
        for name in names
    }


def _planted(root: Path, tree: Mapping[str, tuple[bytes, str]]) -> dict[str, str]:
    """Write each file of `tree` under `root/run`, answering the digests a lake would send.

    A `held` file is sent with its own digest, an `other` one with another's, an `absent` one not.
    """
    for name, (payload, _) in tree.items():
        (root / "run" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "run" / name).write_bytes(payload)
    return {
        f"run/{name}": hashlib.sha256(payload if label == "held" else payload + b"!").hexdigest()
        for name, (payload, label) in tree.items()
        if label != "absent"
    }


@pytest.fixture
def ssh() -> _Host:
    return _Host()


@pytest.fixture
def collector(kept: Evidence, stored: Callable[[str, bytes], Path], ssh: _Host) -> Collector:
    """A collector reaching `ssh`, over a center whose lake holds `_CENTER`."""
    for name, payload in _CENTER.items():
        stored(name, payload)
    return Collector(kept.root, transport=ssh)


@pytest.fixture
def host(tmp_path: Path) -> Path:
    """A job host's workspace: `_HOST`, a link to the run's `held.bin` and an empty folder."""
    root = tmp_path / "host"
    for name, payload in _HOST.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(payload)
    (root / _RUN / "link.bin").symlink_to(root / _RUN / "held.bin")
    (root / _RUN / "empty").mkdir()
    return root


@pytest.fixture
def evicted(collector: Collector, host: Path) -> Eviction:
    """What evicting the run from `host` did."""
    return collector.evict("gold", root=str(host), path=_RUN, python="python")


def test_a_host_eviction_removes_only_the_files_the_lake_holds_intact_at_their_path(
    evicted: Eviction, collector: Collector, host: Path
) -> None:
    assert sorted(evicted.removed) == [f"{_RUN}/deep/a/b/held.txt", f"{_RUN}/held.bin"]
    assert _left(host) == {
        "outside.bin",
        f"{_RUN}/other.bin",
        f"{_RUN}/new.bin",
        f"{_RUN}/elsewhere.bin",
        f"{_RUN}/link.bin",
        collector.digests,
    }


def test_a_host_eviction_says_why_each_file_it_left_in_place_stayed(evicted: Eviction) -> None:
    assert evicted.kept == {
        f"{_RUN}/elsewhere.bin": _UNINDEXED,
        f"{_RUN}/link.bin": "a link or special file, never followed",
        f"{_RUN}/new.bin": _UNINDEXED,
        f"{_RUN}/other.bin": "the lake holds other bytes at this path",
    }


def test_a_host_eviction_removes_only_the_folders_its_removals_emptied(
    evicted: Eviction, host: Path
) -> None:
    assert evicted.emptied == [f"{_RUN}/deep/a/b", f"{_RUN}/deep/a", f"{_RUN}/deep"]
    assert (host / _RUN / "empty").is_dir()


def test_a_host_eviction_drops_the_cached_digests_of_the_files_it_removed(
    evicted: Eviction, collector: Collector, host: Path
) -> None:
    cached = json.loads((host / collector.digests).read_text(encoding="utf-8"))
    assert set(cached) == {f"{_RUN}/other.bin"}


def test_a_lake_object_that_reads_back_damaged_keeps_the_host_copy(
    collector: Collector,
    host: Path,
    harm: Callable[..., None],
    digest_of: Callable[[bytes], str],
) -> None:
    harm(f"DELETE FROM lake.blobs WHERE sha256 = '{digest_of(b'held')}'")
    eviction = collector.evict("gold", root=str(host), path=_RUN, python="python")
    assert eviction.kept[f"{_RUN}/held.bin"] == _UNINDEXED
    assert (host / _RUN / "held.bin").read_bytes() == b"held"


@pytest.mark.parametrize(
    "path",
    [
        pytest.param("", id="nothing-named"),
        pytest.param(".", id="the-workspace-itself"),
        pytest.param("/", id="the-filesystem-root"),
        pytest.param("..", id="the-workspace-parent"),
        pytest.param("../escape", id="a-sibling-of-the-workspace"),
        pytest.param("/etc", id="an-absolute-path"),
        pytest.param(f"{_RUN}/../run", id="a-path-climbing-back-in"),
    ],
)
def test_a_host_eviction_refuses_the_workspace_and_any_path_outside_it_before_reaching_the_host(
    collector: Collector, ssh: _Host, host: Path, path: str
) -> None:
    with pytest.raises(ValueError, match="project-relative"):
        collector.evict("gold", root=str(host), path=path, python="python")
    assert ssh.scripts == []


@pytest.mark.parametrize(
    "relative",
    [
        pytest.param("linked/run", id="a-folder-reached-through-a-link"),
        pytest.param(f"{_RUN}/link.bin", id="a-link-named-itself"),
    ],
)
def test_the_host_refuses_a_path_reached_through_a_link_and_removes_nothing(
    host: Path, relative: str
) -> None:
    (host / "linked").symlink_to(host / "results")
    before = _left(host)
    held = {name: hashlib.sha256(payload).hexdigest() for name, payload in _HOST.items()}
    with pytest.raises(ValueError, match="through a link"):
        evict(str(host), relative=relative, center=Center(held))
    assert _left(host) == before


_TREES = st.dictionaries(
    st.tuples(
        st.sampled_from(["", "dir/", "dir/sub/"]), st.text("abc", min_size=1, max_size=3)
    ).map("".join),
    st.tuples(st.binary(max_size=8), st.sampled_from(["held", "other", "absent"])),
    max_size=8,
)


@given(tree=_TREES)
def test_whatever_the_tree_a_host_keeps_every_file_the_lake_does_not_hold_at_its_path(
    tree: dict[str, tuple[bytes, str]],
) -> None:
    with TemporaryDirectory() as folder:
        kept = _planted(Path(folder), tree)
        with redirect_stdout(io.StringIO()):
            evict(folder, relative="run", center=Center(kept))
        left = _left(Path(folder))
    assert left == {f"run/{name}" for name, (_, label) in tree.items() if label != "held"}
