import os
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest

from mainboard.dispatch import HostUnreachable
from mainboard.dispatch.agent import Agent, AgentRefused, Rules, Scope
from mainboard.dispatch.provenance import Row, Status, blob_of
from mainboard.dispatch.provenance import listing as listed
from mainboard.dispatch.snapshots import (
    CLOSURE,
    STAMP,
    WRAPPERS,
    Mirrored,
    Sealed,
    Snapshots,
    stamped,
    writable,
)
from mainboard.dispatch.sync import compiled

from .support import InProcessLink, RecordingAgent, links_on_this_host


def mirrored(*sources: str, ignore: tuple[str, ...] = (), deny: tuple[str, ...] = ()) -> Mirrored:
    """The image a command that ships the mirror pins: the synced scope, filled back."""
    scope = Scope(sources, ignore=Rules({"": compiled(ignore)}), deny=Rules({"": compiled(deny)}))
    return Mirrored(scope=scope.spec())


def local() -> Agent:
    """The agent of a host that is this machine, answering in this process."""
    return Agent(InProcessLink())


def test_a_snapshots_stamp_records_the_commit_and_digest_it_was_dispatched_from() -> None:
    """A dispatched job's workspace is a mirror with no history, so this file is the history.

    The key stays alone on the first line, which is what it has always been and what every stamp
    already on every host holds; the provenance follows it as named lines.
    """
    stamp = stamped("e975499", commit="e975499f" * 5, digest="9a" * 32)

    assert stamp.splitlines() == ["e975499", f"commit {'e975499f' * 5}", f"digest {'9a' * 32}"]
    assert stamp.endswith("\n")
    # A workspace git answered nothing for writes the key and nothing it cannot stand behind.
    assert stamped("untracked", commit="", digest="") == "untracked\n"


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


def test_a_pin_asks_the_host_for_one_tree_named_by_its_key_and_answers_where_it_stands() -> None:
    """Everything the host needs is resolved here, down to the paths and the stamp it writes."""
    agent = RecordingAgent()
    image = mirrored("research/compression", "mainboard.toml")
    pinned = Snapshots("/work/projects/").pin(
        agent,
        key="abc1234",
        image=image,
        results="./research/compression/raw",
        commit="c0ffee",
        digest="d1",
    )
    assert pinned == "/work/projects/.mainboard/dispatch/sources/abc1234"
    [request] = agent.requests
    asked = request["pin"]
    assert asked["root"] == "/work/projects"
    assert asked["base"] == "/work/projects/.mainboard/dispatch/sources"
    assert asked["stamp"] == "abc1234\ncommit c0ffee\ndigest d1\n"
    assert asked["results"] == "research/compression/raw"
    assert asked["image"] == {"kind": "mirrored", "scope": image.scope}
    assert (asked["script"], asked["wrapper"], asked["prefix"]) == ("", "", "")


def test_a_sealed_pin_sends_its_needs_normalised_and_every_live_path_it_links() -> None:
    agent = RecordingAgent()
    staged = f".mainboard/dispatch/jobs/job-{'c' * 64}.sh"
    Snapshots("/work/projects").pin(
        agent,
        key="abc1234-9f9f9f9f",
        image=Sealed(listing="jobs/closure.tsv", needs=("./data/corpus",), pins=("p/x",)),
        results="research/node/evidence",
        digest="ab" * 32,
        script=staged,
    )
    asked = agent.requests[0]["pin"]
    assert asked["image"] == {
        "kind": "sealed",
        "listing": "jobs/closure.tsv",
        "digest": "ab" * 32,
        "needs": ["data/corpus"],
        "pins": ["p/x"],
        "staging": ".mainboard/pins",
        "live": ["data/corpus", "research/node/evidence"],
    }
    assert (asked["script"], asked["wrapper"]) == (staged, f"{WRAPPERS}/job-{'c' * 64}.sh")


@pytest.mark.parametrize(
    ("answer", "raised", "match"),
    [
        (AgentRefused("missing source: src"), SystemExit, "could not pin the source tree"),
        (HostUnreachable("ssh: connect to host gold port 22: timed out"), HostUnreachable, "22"),
    ],
)
def test_a_host_that_refused_or_dropped_while_pinning_is_told_apart(
    answer: BaseException, raised: type[BaseException], match: str
) -> None:
    """A tree the host could not build refuses the dispatch; a dropped host is retried."""
    with pytest.raises(raised, match=match):
        Snapshots("/work/projects").pin(
            RecordingAgent(answer), key="abc1234", image=mirrored("src")
        )


_SEALED = Sealed(listing=".mainboard/dispatch/jobs/closure-abc.tsv")


@pytest.mark.parametrize(
    ("fields", "refusal"),
    [
        pytest.param({"key": "a/b"}, "one directory name", id="a-key-that-is-a-path"),
        pytest.param(
            {"image": _SEALED, "digest": "ab"},
            "complete closure digest",
            id="a-truncated-closure-digest",
        ),
        pytest.param(
            {"image": _SEALED.model_copy(update={"needs": ("/etc",)}), "digest": "ab" * 32},
            "relative paths below the root",
            id="a-need-outside-the-workspace",
        ),
        pytest.param(
            {"image": _SEALED, "digest": "ab" * 32, "results": f"{WRAPPERS}/x"},
            "relative paths below the root",
            id="sealed-results-over-the-frozen-wrappers",
        ),
        pytest.param(
            {"results": CLOSURE},
            "must not replace snapshot control files",
            id="mirrored-results-over-the-closure",
        ),
        pytest.param(
            {"script": ".mainboard/dispatch/jobs/job-abc.sh"},
            "full SHA-256 name",
            id="a-wrapper-not-named-by-its-whole-digest",
        ),
    ],
)
def test_a_pin_refuses_a_path_that_could_replace_its_own_controls_before_reaching_the_host(
    fields: dict[str, str | Sealed], refusal: str
) -> None:
    """Every one of these would have the snapshot shell link or remove over what it verifies."""
    agent = RecordingAgent()
    with pytest.raises(ValueError, match=refusal):
        Snapshots("/work/projects").pin(
            agent, **{"key": "abc1234", "image": mirrored("src"), **fields}
        )
    assert agent.requests == []


def _mirror(root: Path) -> None:
    """A host mirror as a sync leaves one: shipped source, a data dir, an env, host artifacts."""
    (root / "research/compression/pkg").mkdir(parents=True)
    (root / "research/compression/pkg/mod.py").write_text("v1\n", encoding="utf-8")
    (root / "research/compression/pkg/spare.py").write_text("spare\n", encoding="utf-8")
    (root / "research/compression/.gitignore").write_text("raw/\n", encoding="utf-8")
    (root / "research/compression/raw").mkdir()
    (root / "research/compression/raw/earlier.json").write_text("{}\n", encoding="utf-8")
    (root / "research/data").mkdir()
    (root / "research/data/corpus.txt").write_text("corpus\n", encoding="utf-8")
    (root / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (root / ".mainboard/envs/default/.pixi/envs/default").mkdir(parents=True)
    (root / ".mainboard/envs/default/.pixi/envs/default/marker").write_text(
        "env\n", encoding="utf-8"
    )
    (root / ".mainboard/envs/default/pixi.toml").write_text("[workspace]\n", encoding="utf-8")
    (root / ".mainboard/dispatch/logs").mkdir(parents=True)
    (root / ".mainboard/dispatch/jobs").mkdir(parents=True)
    (root / ".mainboard/vendor/house/src/house").mkdir(parents=True)
    (root / ".mainboard/vendor/house/src/house/__init__.py").write_text("", encoding="utf-8")
    (root / ".mainboard/vendor/house/src/house/extra.py").write_text("", encoding="utf-8")


@pytest.fixture
def sealed_mirror(tmp_path: Path) -> tuple[Snapshots, Sealed, str]:
    root = tmp_path / "mirror with spaces"
    _mirror(root)
    payload = listed(
        [
            Row(path=path, blob=blob_of(root / path), status=status)
            for path, status in (
                ("research/compression/pkg/mod.py", Status.CLEAN),
                ("research/compression/pkg/spare.py", Status.BUILT),
            )
        ]
    ).encode()
    listing = ".mainboard/dispatch/jobs/closure-source.tsv"
    (root / listing).write_bytes(payload)
    return Snapshots(str(root)), Sealed(listing=listing), sha256(payload).hexdigest()


@links_on_this_host
def test_parallel_pins_freeze_the_listing_and_survive_mirror_replacement(
    sealed_mirror: tuple[Snapshots, Sealed, str],
) -> None:
    trees, image, digest = sealed_mirror
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [
            workers.submit(trees.pin, local(), key="same", image=image, digest=digest)
            for _ in range(2)
        ]
        paths = [future.result() for future in futures]
    assert paths[0] == paths[1]
    frozen = Path(paths[0])
    root = Path(trees.root)
    assert (frozen / CLOSURE).read_bytes() == (root / image.listing).read_bytes()
    assert not (frozen / CLOSURE).is_symlink()
    replacement = root / "replacement.py"
    replacement.write_text("v2\n")
    replacement.replace(root / "research/compression/pkg/mod.py")
    (root / image.listing).write_text("later listing\n")
    assert trees.pin(local(), key="same", image=image, digest=digest) == str(frozen)
    assert (frozen / "research/compression/pkg/mod.py").read_text() == "v1\n"
    assert not list(Path(trees.base).glob(".pending.*"))


@pytest.mark.parametrize("corruption", ["listing", "clean", "built", "missing"])
def test_wrong_mirror_bytes_never_publish_a_snapshot(
    sealed_mirror: tuple[Snapshots, Sealed, str],
    corruption: str,
) -> None:
    trees, image, digest = sealed_mirror
    root = Path(trees.root)
    path = (
        root
        / {
            "listing": image.listing,
            "clean": "research/compression/pkg/mod.py",
            "built": "research/compression/pkg/spare.py",
            "missing": "research/compression/pkg/mod.py",
        }[corruption]
    )
    if corruption == "missing":
        path.unlink()
    else:
        path.write_text("changed\n")
    with pytest.raises(SystemExit, match="could not pin"):
        trees.pin(local(), key="wrong", image=image, digest=digest)
    assert not Path(trees.path("wrong")).exists()
    assert not list(Path(trees.base).glob(".pending.*"))


@links_on_this_host
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("field", ["needs", "results"])
def test_live_paths_cannot_replace_a_verified_source_directory(
    sealed_mirror: tuple[Snapshots, Sealed, str],
    reuse: bool,
    field: str,
) -> None:
    trees, image, digest = sealed_mirror
    if reuse:
        trees.pin(local(), key="overlap", image=image, digest=digest)
    kwargs = {"results": "./research/compression/pkg"} if field == "results" else {}
    if field == "needs":
        image = image.model_copy(update={"needs": ("./research/compression/pkg",)})
    with pytest.raises(SystemExit, match="live path overlaps source"):
        trees.pin(local(), key="overlap", image=image, digest=digest, **kwargs)
    assert (Path(trees.root) / "research/compression/pkg/mod.py").read_text() == "v1\n"
    assert Path(trees.path("overlap")).exists() is reuse


@links_on_this_host
def test_wrappers_are_frozen_by_bytes_and_not_repaired_after_corruption(
    sealed_mirror: tuple[Snapshots, Sealed, str],
) -> None:
    trees, image, digest = sealed_mirror
    root = Path(trees.root)
    frozen = Path(trees.pin(local(), key="wrappers", image=image, digest=digest))
    payloads = (b"#!/bin/sh\r\n# non-UTF8: \xff\r\n", b"#!/bin/sh\n# second\n")
    for payload in payloads:
        staged = f".mainboard/dispatch/jobs/job-{sha256(payload).hexdigest()}.sh"
        (root / staged).write_bytes(payload)
        (root / staged).chmod(0o640)
        trees.pin(local(), key="wrappers", image=image, digest=digest, script=staged)
        wrapper = frozen / Snapshots.script(staged)
        assert wrapper.read_bytes() == payload and wrapper.stat().st_mode & 0o777 == 0o640
        assert not wrapper.is_symlink() and not (frozen / WRAPPERS).is_symlink()
        (root / staged).write_text("changed mirror wrapper\n")
        trees.pin(local(), key="wrappers", image=image, digest=digest, script=staged)
        assert wrapper.read_bytes() == payload
    assert len(list((frozen / WRAPPERS).glob("job-*.sh"))) == 2
    with pytest.raises(SystemExit, match="wrapper digest mismatch"):
        trees.pin(local(), key="bad-wrapper", image=image, digest=digest, script=staged)
    assert not Path(trees.path("bad-wrapper")).exists()
    wrapper.write_text("corrupt frozen wrapper\n")
    with pytest.raises(SystemExit, match="wrapper digest mismatch"):
        trees.pin(local(), key="wrappers", image=image, digest=digest, script=staged)
    assert wrapper.read_text() == "corrupt frozen wrapper\n"


@links_on_this_host
def test_a_pinned_tree_survives_the_sync_that_rewrites_the_mirror_under_it(
    tmp_path: Path,
) -> None:
    """The fault itself, run for real: a mirror sync must not reach the code of a live job."""
    root = tmp_path / "projects"
    _mirror(root)
    pinned = Path(
        Snapshots(str(root)).pin(
            local(),
            key="abc1234",
            image=mirrored("research/compression", ignore=("raw/",), deny=(".mainboard/",)),
            results="research/compression/raw",
        )
    )
    frozen = pinned / "research/compression/pkg/mod.py"
    assert frozen.stat().st_ino == (root / "research/compression/pkg/mod.py").stat().st_ino
    # The environment and the data directory are reached live, and the results path points back
    # at the mirror, which is where the pull already looks.
    assert (pinned / ".mainboard/envs/default/.pixi").is_symlink()
    assert (pinned / ".mainboard/dispatch").is_symlink()
    marker = pinned / ".mainboard/envs/default/.pixi/envs/default/marker"
    assert marker.read_text(encoding="utf-8") == "env\n"
    # The generated manifest is hardlinked rather than symlinked, so the job's own tooling
    # recompiles it into this snapshot instead of over the mirror's copy.
    generated = pinned / ".mainboard/envs/default/pixi.toml"
    assert not generated.is_symlink()
    assert generated.stat().st_ino == (root / ".mainboard/envs/default/pixi.toml").stat().st_ino
    assert (pinned / "research/data").is_symlink()
    assert (pinned / "research/compression/raw").resolve() == root / "research/compression/raw"

    # The mirror writes a changed file beside its name and renames it over the old one.
    later = root / "research/compression/pkg/.mod.py.pending"
    later.write_text("v2\n", encoding="utf-8")
    later.replace(root / "research/compression/pkg/mod.py")
    assert (root / "research/compression/pkg/mod.py").read_text(encoding="utf-8") == "v2\n"
    assert frozen.read_text(encoding="utf-8") == "v1\n"


@links_on_this_host
def test_a_sealed_tree_holds_the_listed_files_the_environment_and_the_needs_and_nothing_else(
    tmp_path: Path,
) -> None:
    """A job's tree reaches the mirror through what it declared and through nothing else.

    The mirror keeps a spare module beside the shipped one, a data directory beside the code
    and a vendored tree under the generated directory; the snapshot holds the listed files by
    hardlink, links the one need and the results path back, hands the environment through and
    leaves everything else unreachable, so an import the closure missed fails on the node.
    """
    root = tmp_path / "projects"
    _mirror(root)
    listing = ".mainboard/dispatch/jobs/closure-abc.tsv"
    payload = listed(
        [
            Row(path=path, blob=blob_of(root / path), status=Status.CLEAN)
            for path in (
                "research/compression/pkg/mod.py",
                ".mainboard/vendor/house/src/house/__init__.py",
            )
        ]
    ).encode()
    (root / listing).write_bytes(payload)
    digest = sha256(payload).hexdigest()
    pinned = Path(
        Snapshots(str(root)).pin(
            local(),
            key="abc1234-9f9f9f9f",
            image=Sealed(listing=listing, needs=("research/data",)),
            results="research/compression/raw",
            digest=digest,
        )
    )
    frozen = pinned / "research/compression/pkg/mod.py"
    assert frozen.stat().st_ino == (root / "research/compression/pkg/mod.py").stat().st_ino
    assert not (pinned / "research/compression/pkg/spare.py").exists()
    assert (pinned / ".mainboard/vendor/house/src/house/__init__.py").is_file()
    assert not (pinned / ".mainboard/vendor/house/src/house/extra.py").exists()
    assert not (pinned / ".mainboard/vendor").is_symlink()
    assert (pinned / "research/data").is_symlink()
    assert (pinned / "research/data/corpus.txt").read_text(encoding="utf-8") == "corpus\n"
    assert (pinned / "research/compression/raw").resolve() == root / "research/compression/raw"
    assert (pinned / ".mainboard/envs/default/.pixi").is_symlink()
    assert (pinned / ".mainboard/dispatch").is_symlink()
    assert not (pinned / ".gitignore").exists()
    assert sorted(entry.name for entry in pinned.iterdir()) == [
        ".mainboard",
        CLOSURE,
        STAMP,
        "research",
    ]
    assert (pinned / CLOSURE).read_bytes() == payload
    # A need the mirror does not hold refuses the dispatch by name rather than dangling.
    with pytest.raises(SystemExit, match="the need research/absent is not on the mirror"):
        Snapshots(str(root)).pin(
            local(),
            key="abc1234-9f9f9f9f",
            image=Sealed(listing=listing, needs=("research/absent",)),
            digest=digest,
        )


@links_on_this_host
def test_a_second_dispatch_of_one_tree_reuses_the_snapshot_instead_of_rebuilding_it(
    tmp_path: Path,
) -> None:
    """Thirty five jobs off one commit pay for one tree, and none of them waits for a rebuild."""
    root = tmp_path / "projects"
    _mirror(root)
    snapshots = Snapshots(str(root))
    image = mirrored("research/compression")
    pinned = Path(snapshots.pin(local(), key="abc1234", image=image))
    (pinned / "research/compression/pkg/mod.py").unlink()
    assert snapshots.pin(local(), key="abc1234", image=image) == str(pinned)
    assert not (pinned / "research/compression/pkg/mod.py").exists()


@links_on_this_host
def test_two_batches_off_one_commit_each_get_their_own_results_link(tmp_path: Path) -> None:
    """A snapshot is keyed on the source and a results path is not part of the source.

    Two batches dispatched from one commit therefore share a tree while declaring different
    results paths, and the second one used to get no link at all: its pull failed on a path that
    did not exist while its receipts sat inside the snapshot (gh200-closure reusing
    gh200-directed-tree's tree, 2026-09-05).
    """
    root = tmp_path / "projects"
    _mirror(root)
    (root / "research/compression/evidence").mkdir()
    snapshots = Snapshots(str(root))
    image = mirrored("research/compression")

    first = Path(
        snapshots.pin(local(), key="abc1234", image=image, results="research/compression/raw")
    )
    second = Path(
        snapshots.pin(local(), key="abc1234", image=image, results="research/compression/evidence")
    )

    assert second == first
    assert (
        first.joinpath("research/compression/raw").resolve() == root / "research/compression/raw"
    )
    assert (
        first.joinpath("research/compression/evidence").resolve()
        == root / "research/compression/evidence"
    )
    # Pinning again with the first path back does not double the link or lose the second.
    snapshots.pin(local(), key="abc1234", image=image, results="research/compression/raw")
    assert first.joinpath("research/compression/raw").is_symlink()
    assert first.joinpath("research/compression/evidence").is_symlink()


@links_on_this_host
def test_a_job_recompiling_its_manifest_writes_into_its_own_tree_not_the_mirrors(
    tmp_path: Path,
) -> None:
    """A generated manifest carries the root it was compiled for, so a pinned tree recompiles one.

    Which means the write has to land here. The mirror's copy is what every other job on the
    host activates through, and the tools that write these files replace them rather than edit
    them, so a hardlink is exactly the right shape: the snapshot's entry moves to its own inode
    and nobody else's job notices.
    """
    root = tmp_path / "projects"
    _mirror(root)
    pinned = Path(
        Snapshots(str(root)).pin(local(), key="abc1234", image=mirrored("research/compression"))
    )
    generated = pinned / ".mainboard/envs/default/pixi.toml"
    mirrored_manifest = root / ".mainboard/envs/default/pixi.toml"
    replacement = generated.with_suffix(".toml.tmp")
    replacement.write_text("[workspace]\nname = 'pinned'\n", encoding="utf-8")
    replacement.replace(generated)
    assert generated.read_text(encoding="utf-8") != mirrored_manifest.read_text(encoding="utf-8")


@links_on_this_host
def test_a_tree_changed_after_its_pin_refuses_every_later_reuse(
    sealed_mirror: tuple[Snapshots, Sealed, str],
) -> None:
    """A source file swapped for a link, or reached through a linked parent, is not the source."""
    trees, image, digest = sealed_mirror
    frozen = Path(trees.pin(local(), key="tampered", image=image, digest=digest))
    module = frozen / "research/compression/pkg/mod.py"
    module.unlink()
    module.symlink_to(Path(trees.root) / "research/compression/pkg/mod.py")
    with pytest.raises(SystemExit, match="missing or linked source"):
        trees.pin(local(), key="tampered", image=image, digest=digest)
    module.unlink()
    package = frozen / "research/compression/pkg"
    package.rename(frozen / "research/compression/elsewhere")
    (frozen / "research/compression/elsewhere/mod.py").write_text("v1\n", encoding="utf-8")
    package.symlink_to(frozen / "research/compression/elsewhere", target_is_directory=True)
    with pytest.raises(SystemExit, match="linked source parent"):
        trees.pin(local(), key="tampered", image=image, digest=digest)
    with pytest.raises(SystemExit, match="snapshot stamp mismatch"):
        trees.pin(local(), key="tampered", image=image, digest=digest, commit="c0ffee")


@links_on_this_host
def test_a_half_built_tree_waits_for_an_operator_rather_than_being_reused(
    sealed_mirror: tuple[Snapshots, Sealed, str],
) -> None:
    trees, image, digest = sealed_mirror
    Path(trees.path("half")).mkdir(parents=True)
    with pytest.raises(SystemExit, match="incomplete snapshot requires inspection"):
        trees.pin(local(), key="half", image=image, digest=digest)


@pytest.mark.parametrize(
    ("row", "refusal"),
    [
        ("../outside.py", "invalid closure path"),
        (".mainboard-jobs/job.sh", "reserved closure path"),
        (CLOSURE, "reserved closure path"),
    ],
)
def test_a_listing_naming_a_path_outside_the_tree_or_its_controls_is_refused(
    tmp_path: Path, row: str, refusal: str
) -> None:
    root = tmp_path / "projects"
    _mirror(root)
    listing = ".mainboard/dispatch/jobs/closure-bad.tsv"
    payload = f"{row}\t{'0' * 64}\tclean\n".encode()
    (root / listing).write_bytes(payload)
    with pytest.raises(SystemExit, match=refusal):
        Snapshots(str(root)).pin(
            local(), key="bad", image=Sealed(listing=listing), digest=sha256(payload).hexdigest()
        )


@links_on_this_host
def test_staged_pins_reach_the_tree_through_their_shared_directory_or_refuse_by_name(
    sealed_mirror: tuple[Snapshots, Sealed, str],
) -> None:
    trees, image, digest = sealed_mirror
    frozen = Path(trees.pin(local(), key="pins", image=image, digest=digest))
    pin = ".mainboard/pins/models--o--n/snapshots/r/tokenizer.json"
    staged = Path(trees.root) / pin
    staged.parent.mkdir(parents=True)
    staged.write_text("{}", encoding="utf-8")
    pinned = image.model_copy(update={"pins": (pin,)})
    trees.pin(local(), key="pins", image=pinned, digest=digest)
    assert (frozen / ".mainboard/pins").is_symlink()
    assert (frozen / pin).read_text(encoding="utf-8") == "{}"
    trees.pin(local(), key="pins", image=pinned, digest=digest)
    absent = image.model_copy(update={"pins": (".mainboard/pins/absent.json",)})
    with pytest.raises(SystemExit, match="the need .mainboard/pins/absent.json is not on"):
        trees.pin(local(), key="pins", image=absent, digest=digest)


@links_on_this_host
def test_a_frozen_wrapper_reached_through_a_link_is_refused(
    sealed_mirror: tuple[Snapshots, Sealed, str],
) -> None:
    trees, image, digest = sealed_mirror
    root = Path(trees.root)
    payload = b"#!/bin/sh\n"
    staged = f".mainboard/dispatch/jobs/job-{sha256(payload).hexdigest()}.sh"
    (root / staged).write_bytes(payload)
    frozen = Path(trees.pin(local(), key="wrap", image=image, digest=digest, script=staged))
    wrapper = frozen / Snapshots.script(staged)
    wrapper.unlink()
    wrapper.symlink_to(root / staged)
    with pytest.raises(SystemExit, match="missing or linked wrapper"):
        trees.pin(local(), key="wrap", image=image, digest=digest, script=staged)
    (frozen / WRAPPERS).rename(frozen / "moved")
    (frozen / WRAPPERS).symlink_to(frozen / "moved", target_is_directory=True)
    with pytest.raises(SystemExit, match="linked wrapper directory"):
        trees.pin(local(), key="wrap", image=image, digest=digest, script=staged)


@links_on_this_host
def test_a_mirrored_tree_carries_links_names_its_prefix_and_leaves_a_linked_results_path_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A results path already reached through a link into the mirror is the mirror's own.

    Clearing it there would clear the mirror's results, so the pin leaves it as it stands.
    Where a file system shares no inode, the copy is a copy.
    """
    root = tmp_path / "projects"
    _mirror(root)
    (root / "research/compression/pkg/alias.py").symlink_to("mod.py")
    (root / ".mainboard/envs/README").write_text("not an environment", encoding="utf-8")
    (root / "research/data/out").mkdir()
    (root / "research/data/out/kept.json").write_text("{}", encoding="utf-8")

    def unshared(*args: str, **kwargs: bool) -> None:
        raise OSError("this file system shares no inodes")

    monkeypatch.setattr(os, "link", unshared)
    prefix = "/prefixes/default/abcd"
    pinned = Path(
        Snapshots(str(root)).pin(
            local(),
            key="abc1234",
            image=mirrored("research/compression"),
            results="research/data/out",
            prefix=prefix,
        )
    )
    assert os.readlink(pinned / "research/compression/pkg/alias.py") == "mod.py"
    assert os.readlink(pinned / ".mainboard/envs/default/.pixi") == f"{prefix}/.pixi"
    assert (root / "research/data/out/kept.json").is_file()
    assert (pinned / "research/data").is_symlink()
    copied = pinned / "research/compression/pkg/mod.py"
    assert copied.stat().st_ino != (root / "research/compression/pkg/mod.py").stat().st_ino


def test_a_mirror_without_its_source_or_generated_tree_pins_what_it_has_or_refuses(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bare"
    (root / "src").mkdir(parents=True)
    (root / "src/run.py").write_text("print()\n", encoding="utf-8")
    pinned = Path(Snapshots(str(root)).pin(local(), key="bare", image=mirrored("src")))
    assert (pinned / "src/run.py").is_file()
    with pytest.raises(SystemExit, match="missing source: gone"):
        Snapshots(str(root)).pin(local(), key="gone", image=mirrored("gone"))
