"""B05: nothing a queued, held, PBS, offline or historical job still names is pruned.

Every witness runs in a temporary workspace against fake hosts: no scheduler, lake or network is
reached. A tree or prefix an inventory leaves unaccounted for is kept, never assumed unused. Pins
protect the roots a host already has, `dispatch/sources` and `prefixes/<environment>`, in place.
"""

import hashlib
import json
import os
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mainboard.board import Board
from mainboard.core import Project
from mainboard.core.retention import Retention
from mainboard.dispatch import Snapshots
from mainboard.dispatch.agent import program
from mainboard.dispatch.snapshots import Sealed, queued_here, stamped
from mainboard.dispatch.spool import Claim, Spool, Submission, spool_path
from mainboard.dispatch.state import RunRecord
from mainboard.engines.compile import Prefixes
from mainboard.manifest import Header, Manifest

_PAYLOAD = b"bytes a job may still read\n"

# Seconds since each tree and prefix was last used. Two are past the retention window and outside
# the newest few; the rest are what a running wave keeps.
_AGES = {"old": 9 * 86400, "stale": 9 * 86400, "new1": 3, "new2": 2, "new3": 1}


class Host:
    """A host workspace holding aged source trees and prefixes, none of them pinned.

    listing: what pueue answers about its queue, None for a host without pueue.
    runs: what the center's registry holds about runs on this host.
    """

    def __init__(self, root: Path) -> None:
        (root / "mainboard.toml").write_text('[workspace]\nname = "retention"\n', encoding="utf-8")
        self.root = root
        self.out = Project().out(root)
        self.snapshots = Snapshots(str(root))
        self.prefixes = Prefixes(root, Manifest(workspace=Header(name="retention")))
        self.spool = Spool(Path(spool_path(str(root))))
        self.listing: str | None = '{"tasks": {}}'
        self.runs: list[RunRecord] = []
        for name in _AGES:
            for directory in (self.tree(name), self.prefixes.path(name)):
                directory.mkdir(parents=True)
                (directory / "bytes").write_bytes(_PAYLOAD)
            self._age(name)

    @staticmethod
    def expire(path: Path, seconds: float) -> None:
        """Make `path` last used `seconds` ago."""
        stamp = time.time() - seconds
        os.utime(path, (stamp, stamp))

    def activate(self, name: str) -> None:
        """Link tree `name` to the prefix of the same name, as a pin with an environment does."""
        link = self.tree(name) / self.out.name / "envs/default/.pixi"
        link.parent.mkdir(parents=True)
        link.symlink_to(self.prefixes.path(name) / ".pixi")
        self._age(name)

    def bystanders(self) -> list[Path]:
        """Age the prefixes of two other environments, which no sweep of this one may touch."""
        roots = [
            Prefixes(self.root, self.prefixes.manifest, environment).path(name)
            for environment in ("historical", "ancient")
            for name in ("old", "stale", "new1")
        ]
        for root in roots:
            root.mkdir(parents=True)
            self.expire(root, 9 * 86400)
        return roots

    def freeze(self) -> tuple[Path, Path]:
        """Queue a job before retention: its built prefix, and the tree whose link names it."""
        frozen = self.prefixes.path("frozen")
        frozen.mkdir(parents=True)
        for name in ("pixi.toml", "pixi.lock", "activate.sh"):
            (frozen / name).write_bytes(_PAYLOAD)
        (frozen / Project().marker("prefix")).write_text("frozen\n", encoding="utf-8")
        source = self.tree("queued") / self.out.name / "envs/default"
        source.mkdir(parents=True)
        (source / ".pixi").symlink_to(frozen / ".pixi")
        return frozen, source

    def inventory(self) -> Retention:
        """Take the inventory of everything queued here and registered."""
        with self._pueue():
            return self.snapshots.inventory(busy=queued_here(), registry=self.runs)

    def pinned(self, prefix: Path) -> Path:
        """Have the agent pin one sealed job activating `prefix`, queued ten days ago."""
        (self.root / "job.py").write_bytes(b"pass\n")
        listing = f"job.py\t{hashlib.sha256(b'pass\n').hexdigest()}\tsource\n"
        (self.root / "closure.tsv").write_text(listing, encoding="utf-8")
        digest = hashlib.sha256(listing.encode()).hexdigest()
        spec = program.PinSpec(
            root=str(self.root),
            base=self.snapshots.base,
            key="pinned",
            stamp=stamped("pinned", commit="", digest=digest),
            out=self.out.name,
            prefix=str(prefix),
            environment="default",
            results="",
            script="",
            wrapper="",
            image=Sealed(listing="closure.tsv").request(digest, ""),
        )
        tree = Path(program.Snapshot(spec).pinned())
        self.expire(tree, 10 * 86400)
        return tree

    def provide(self) -> None:
        """Run `Board.provide` as a job start does, finding what is queued here."""
        plan = SimpleNamespace(env="default", profile=SimpleNamespace(modules={}))
        with (
            self._pueue(),
            patch.object(Board, "plan", return_value=plan),
            patch.object(Prefixes, "materialize", return_value=self.prefixes.path("new3")),
        ):
            Board(self.root).provide()

    def queued(self) -> str | None:
        """Ask the host's pueue what it lists."""
        with self._pueue():
            return queued_here()

    def survives(self, *names: str) -> bool:
        """Say whether the trees and prefixes called `names` still hold their exact bytes."""
        return all(
            (directory / "bytes").read_bytes() == _PAYLOAD
            for name in names
            for directory in (self.tree(name), self.prefixes.path(name))
        )

    def tree(self, name: str) -> Path:
        return Path(self.snapshots.path(name))

    def _age(self, name: str) -> None:
        """Set tree and prefix `name` back to their age, which any entry added inside undid."""
        for directory in (self.tree(name), self.prefixes.path(name)):
            self.expire(directory, _AGES[name])

    @contextmanager
    def _pueue(self) -> Generator[None]:
        """Stand in for the host's pueue daemon."""
        with (
            patch(
                "mainboard.dispatch.snapshots.shutil.which",
                return_value=None if self.listing is None else "pueue",
            ),
            patch(
                "mainboard.dispatch.snapshots.subprocess.run",
                return_value=SimpleNamespace(stdout=self.listing),
            ),
        ):
            yield


def hold(host: Host, name: str, *, claimed: bool = False) -> None:
    """Queue a held-line submission that runs from tree `name`, optionally claimed."""
    job = Submission(
        handle="h01",
        label="t",
        script="job.sh",
        cwd=str(host.tree(name)),
        walltime="00:00:05",
        submitted="now",
    )
    host.spool.publish(host.spool.inbox, "h01.json", text=job.model_dump_json())
    if claimed:
        host.spool.claim(job, Claim(pid=123))


def queue(host: Host, state: str, tree: Path, *, wrapper: bool = True) -> None:
    """Queue a pueue task in `state` that runs from `tree`.

    The tree is named by the wrapper script's text, whose path has spaces in it, or else by the
    task's working directory beside a wrapper that no longer exists.
    """
    script = host.root / "job with spaces.sh"
    if wrapper:
        script.write_text(f'cd "{tree}"\n', encoding="utf-8")
    task = {
        "status": {state: {}},
        "path": str(host.root if wrapper else tree),
        "command": f"bash '{script}'",
    }
    host.listing = json.dumps({"tasks": {"1": task}})


def register(host: Host, source: str, verdict: str | None) -> None:
    """Record a run on a host the center cannot reach, whose handle may never be reconciled."""
    host.runs.append(
        RunRecord(
            handle="lost",
            target="offline",
            kind="pbs",
            source=source,
            script="job.sh",
            args="",
            submitted_at="now",
            verdict=verdict,
        )
    )


def stage(host: Host, tree: Path) -> None:
    """Stage the job script of a PBS job waiting for its allocation to run from `tree`."""
    script = Path(host.snapshots.base).parent / "jobs" / "job-staged.sh"
    script.parent.mkdir(parents=True)
    script.write_text(f'cd "{tree}"\n', encoding="utf-8")


def pin(host: Host, name: str, content: bytes) -> None:
    """Publish `content` as the host pin `name`."""
    pins = Path(host.snapshots.base) / ".retention"
    pins.mkdir(exist_ok=True)
    (pins / name).write_bytes(content)


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Host:
    # The workspace this process serves is the host's, so its state directory is the one the
    # spool and the prefixes both name, whatever names the installed tool answers to.
    monkeypatch.chdir(tmp_path)
    return Host(tmp_path)


def test_provide_keeps_what_a_held_job_names(host: Host) -> None:
    """The B05 reproduction at the call site that pruned: a job start finding an empty queue."""
    hold(host, "old")
    host.activate("old")
    host.provide()
    assert host.survives("old", "new3")
    assert not host.tree("stale").exists()
    assert not host.prefixes.path("stale").exists()


def test_provide_without_pueue_deletes_nothing(host: Host) -> None:
    """A queue that cannot be asked is unknown, not empty: every root stays."""
    hold(host, "old")
    host.listing = None
    host.provide()
    assert host.survives(*_AGES)


@pytest.mark.parametrize(
    ("names", "pruned"),
    [
        pytest.param(lambda host: None, ["old", "stale"], id="nothing"),
        pytest.param(lambda host: hold(host, "old"), ["stale"], id="held"),
        pytest.param(lambda host: hold(host, "old", claimed=True), ["stale"], id="claimed"),
        pytest.param(lambda h: queue(h, "Queued", h.tree("old")), ["stale"], id="pueue-queued"),
        pytest.param(lambda h: queue(h, "Running", h.tree("old")), ["stale"], id="pueue-running"),
        pytest.param(lambda h: queue(h, "Done", h.tree("old")), ["stale"], id="pueue-done"),
        pytest.param(
            lambda h: queue(h, "Done", h.tree("old"), wrapper=False), ["stale"], id="pueue-cwd"
        ),
        pytest.param(lambda host: register(host, "old", None), ["stale"], id="run-unreconciled"),
        pytest.param(lambda host: register(host, "old", "running"), ["stale"], id="run-running"),
        pytest.param(lambda host: register(host, "old", "ok"), ["stale"], id="run-settled"),
        pytest.param(lambda h: stage(h, h.tree("old")), ["stale"], id="staged-script"),
    ],
)
def test_whatever_names_a_tree_keeps_it(
    host: Host, names: Callable[[Host], None], pruned: list[str]
) -> None:
    names(host)
    inventory = host.inventory()
    assert not inventory.unknown
    assert sorted(host.snapshots.prune(inventory)) == pruned
    assert host.snapshots.prune(inventory) == []
    assert host.survives(*_AGES.keys() - set(pruned))


@pytest.mark.parametrize(
    "gap",
    [
        pytest.param(lambda host: setattr(host, "listing", None), id="scheduler"),
        pytest.param(
            lambda host: host.spool.publish(host.spool.inbox, "h01.json", text="{"),
            id="torn-submission",
        ),
        pytest.param(
            lambda host: host.spool.publish(
                host.spool.claims, "h01.json", text=Claim().model_dump_json()
            ),
            id="claim-without-submission",
        ),
        pytest.param(lambda host: pin(host, "torn.json", b"{"), id="torn-pin"),
        pytest.param(
            lambda host: pin(host, "0.json", b'{"sources": ["/x"]}'), id="pin-not-by-content"
        ),
    ],
)
def test_an_unknown_inventory_removes_nothing(
    host: Host, gap: Callable[[Host], None | bool]
) -> None:
    gap(host)
    inventory = host.inventory()
    assert host.snapshots.prune(inventory) == []
    assert host.prefixes.prune(live=(), inventory=inventory) == []
    assert host.survives(*_AGES)


def test_a_pin_outlives_a_restart_with_the_center_offline(host: Host) -> None:
    pinned = host.pinned(host.prefixes.path("old"))
    # A fresh process and an authority that knows nothing: only the host's own pins speak.
    restarted = Snapshots(str(host.root))
    assert sorted(restarted.prune(Retention())) == ["old", "stale"]
    assert (pinned / "job.py").read_bytes() == b"pass\n"
    assert sorted(host.prefixes.prune(live=(), inventory=Retention())) == ["new1", "stale"]
    assert host.survives("new2", "new3")
    # The pin stands beside the trees it protects, with no generation directory in between.
    assert (pinned.parent, host.prefixes.base) == (
        host.out / "dispatch/sources",
        host.out / "prefixes/default",
    )
    assert sorted(path.name for path in host.out.iterdir()) == ["dispatch", "prefixes"]


def test_a_sweep_waits_out_a_pin_in_progress(host: Host) -> None:
    with program.locked(str(Path(host.snapshots.base) / ".pin.lock")):
        assert host.snapshots.prune(Retention()) == []
        assert host.prefixes.prune(live=(), inventory=Retention()) == []
    assert sorted(host.snapshots.prune(Retention())) == ["old", "stale"]


def test_a_prefix_sweep_stays_inside_its_environment(host: Host) -> None:
    bystanders = host.bystanders()
    assert sorted(host.prefixes.prune(live=(), inventory=Retention())) == ["new1", "old", "stale"]
    assert host.prefixes.prune(live=(), inventory=Retention()) == []
    assert all(directory.is_dir() for directory in bystanders)


@pytest.mark.parametrize(
    ("link", "pruned"),
    [
        pytest.param("envs/default", ["new1", "old", "stale"], id="no-link-names-no-prefix"),
        pytest.param("envs/default/.pixi", [], id="unreadable-link-keeps-every-prefix"),
    ],
)
def test_a_tree_environment_link_that_cannot_be_read_keeps_every_prefix(
    host: Host, link: str, pruned: list[str]
) -> None:
    (host.tree("new1") / host.out.name / link).mkdir(parents=True)
    assert sorted(host.prefixes.prune(live=(), inventory=Retention())) == pruned


def test_a_job_queued_before_retention_finds_and_keeps_its_frozen_prefix(host: Host) -> None:
    frozen, source = host.freeze()
    with (
        patch("mainboard.engines.compile.prefixes.digest_of", return_value="frozen"),
        patch("mainboard.engines.compile.prefixes.SecondStage.digest", return_value=""),
    ):
        assert host.prefixes.materialize(source) == frozen
    host.expire(frozen, 9 * 86400)
    assert sorted(host.prefixes.prune(live=(), inventory=Retention())) == ["new1", "old", "stale"]
    assert (frozen / "activate.sh").read_bytes() == _PAYLOAD


def test_aliases_and_nested_roots_keep_the_tree_they_reach(host: Host) -> None:
    alias = host.root / "old-mainboard"
    alias.symlink_to(host.tree("old"), target_is_directory=True)
    nested = host.prefixes.path("old") / ".pixi/envs/default/bin/python"
    roots = [str(alias / "packages/mainboard/src"), str(nested)]
    named = Retention.model_validate({"proofs": roots})
    assert host.snapshots.prune(named) == ["stale"]
    assert sorted(host.prefixes.prune(live=(), inventory=named)) == ["new1", "stale"]
    assert host.survives("old")


@pytest.mark.parametrize(
    "listing",
    [
        pytest.param(None, id="no-pueue"),
        pytest.param("", id="empty"),
        pytest.param("not json", id="not-json"),
        pytest.param('{"tasks": []}', id="tasks-not-a-table"),
        pytest.param('{"tasks": {"1": {"command": "x"}}}', id="task-without-a-directory"),
    ],
)
def test_a_queue_that_cannot_be_read_is_unknown(host: Host, listing: str | None) -> None:
    host.listing = listing
    assert host.queued() is None
