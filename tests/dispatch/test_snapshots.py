from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest
from plumbum import local

from mainboard.dispatch.provenance import Row, Status, blob_of
from mainboard.dispatch.provenance import listing as listed
from mainboard.dispatch.snapshots import (
    CLOSURE,
    STAMP,
    WRAPPERS,
    HostUnreachable,
    Mirrored,
    Sealed,
    Snapshots,
    containers,
    stamped,
    writable,
)

from .support import machine_with, pins_on_this_host


def mirrored(
    *sources: str, filters: tuple[str, ...] = (), exclude: tuple[str, ...] = ()
) -> Mirrored:
    """The image a command that ships the mirror pins: the synced allowlist, filled back."""
    return Mirrored(sources=sources, filters=filters, exclude=exclude)


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


def test_a_pin_writes_that_provenance_into_the_tree_it_freezes() -> None:
    """One write, inside the stamp guard, so the tree says what it is for as long as it stands."""
    remote = machine_with()

    Snapshots("/work/projects").pin(
        remote, key="abc1234", image=mirrored("a"), commit="c0ffee", digest="d1"
    )

    [program] = remote.lines
    stamp = "printf '%s' 'abc1234\ncommit c0ffee\ndigest d1\n'"
    assert f'{stamp} > "$mb_snap/{STAMP}"\nmv -T' in program
    assert f'{stamp} | cmp -s - "$mb_snap/{STAMP}"' in program


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        (["scripts", "mainboard.toml"], ["."]),
        (["research/compression", "research/bale"], [".", "research"]),
        (["a/b/c"], [".", "a", "a/b"]),
    ],
)
def test_every_directory_a_mirrored_snapshot_creates_is_one_it_has_to_fill_from_the_mirror(
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


def test_pinning_the_mirror_copies_the_shipped_set_by_hardlink_and_links_the_rest_back() -> None:
    """A command's bargain: code frozen, environment and data reached live through the mirror."""
    remote = machine_with()
    pinned = Snapshots("/work/projects").pin(
        remote,
        key="abc1234",
        image=mirrored(
            "research/compression", "mainboard.toml", filters=(":- .gitignore",), exclude=(".git",)
        ),
        results="research/compression/raw",
    )
    assert pinned == "/work/projects/.mainboard/dispatch/sources/abc1234"
    [program] = remote.lines
    assert "mb_root=/work/projects" in program
    assert f"mb_final={pinned}" in program
    assert "mb_snap=$(mktemp -d " in program
    assert "--link-dest=/work/projects/" in program
    assert "rsync -aR" in program
    assert 'research/compression mainboard.toml "$mb_snap"/' in program
    # The generated tree is rebuilt as this snapshot's own so a job recompiling its manifest
    # writes here rather than over the description every other job on the host activates through.
    assert 'mkdir -p "$mb_snap"/.mainboard/envs' in program
    assert 'if [ -d "$e" ]; then ln -s "$e" "$mb_snap/$d/$n"; else ln "$e"' in program
    # Every container directory the copy created is then filled from the mirror, which is what
    # puts the data directories and the ancestor ignore files back within the job's reach.
    assert "for d in . research; do" in program
    assert 'ln -s "$e" "$mb_snap/$d/$n"' in program
    # The declared results path is a symlink back to the mirror, so what the job writes there is
    # what the pull already goes looking for, and it is linked after the stamp closes rather
    # than inside it, since the path belongs to this dispatch and the tree does not.
    assert (
        'ln -sfn "$mb_root"/research/compression/raw "$mb_snap"/research/compression/raw'
    ) in program
    stamped_line = f"printf '%s' 'abc1234\n' > \"$mb_snap/{STAMP}\""
    assert stamped_line in program
    assert program.index("ln -sfn") < program.index(stamped_line) < program.rindex("ln -sfn")


def test_pinning_a_closure_copies_exactly_the_listed_files_and_links_only_the_needs() -> None:
    """The bargain a job gets: nothing of the mirror is reachable but what the job declared."""
    remote = machine_with()
    listing = ".mainboard/dispatch/jobs/closure-abc.tsv"
    Snapshots("/work/projects").pin(
        remote,
        key="abc1234-9f9f9f9f",
        image=Sealed(listing=listing, needs=("data/corpus", "data/models/x")),
        results="research/camp/experiments/node/evidence",
        digest="ab" * 32,
    )
    [program] = remote.lines
    assert f'cut -f1 "$mb_snap/{CLOSURE}" | rsync -aL --filter' in program
    assert "--files-from=-" in program
    assert "hide .card.lock" in program and "protect .card.lock" in program
    assert '--link-dest=/work/projects/ ./ "$mb_snap"/' in program
    assert '|| [ "$?" = 24 ]' not in program
    # Only host-local lease rules apply; ordinary ignores cannot alter the exact listing.
    assert program.count("--filter") == 4 and "--exclude" not in program
    assert "for d in" not in program
    # Each need is checked on the mirror and linked in after the stamp, on every dispatch.
    stamp = program.index('mv -T -- "$mb_snap" "$mb_final"')
    for need in ("data/corpus", "data/models/x"):
        check = f'if [ ! -e "$mb_root"/{need} ]; then echo'
        assert program.index(check) < stamp < program.rindex(check)
        assert f"the need {need} is not on the mirror" in program
        assert f'ln -sfn "$mb_root"/{need} "$mb_snap"/{need}' in program
    assert 'mkdir -p "$mb_snap"/data/models' in program
    assert "false; fi" in program and "exit" not in program
    assert program.rindex("ln -sfn") > program.index("ln -sfn")


def test_a_sealed_snapshot_links_its_staged_pins_beside_its_needs() -> None:
    """A Hub pin staged under the workspace is a need: checked on the mirror and linked in."""
    remote = machine_with()
    listing = ".mainboard/dispatch/jobs/closure-abc.tsv"
    Snapshots("/work/projects").pin(
        remote,
        key="abc1234-9f9f9f9f",
        image=Sealed(
            listing=listing,
            needs=("data/corpus",),
            pins=(".mainboard/pins/models--o--n/snapshots/r/tokenizer.json",),
        ),
        results="research/camp/experiments/node/evidence",
        digest="ab" * 32,
    )
    [program] = remote.lines
    assert 'ln -sfn "$mb_root"/data/corpus "$mb_snap"/data/corpus' in program
    pin = ".mainboard/pins/models--o--n/snapshots/r/tokenizer.json"
    assert f"mainboard: the need {pin} is not on the mirror" in program
    assert 'ln -sfn "$mb_root"/.mainboard/pins "$mb_snap"/.mainboard/pins' in program
    assert f'ln -sfn "$mb_root"/{pin}' not in program


def test_pinning_a_tree_the_host_could_not_build_refuses_instead_of_dispatching_into_it() -> None:
    """A job started in a half-built tree imports whatever happened to be copied first."""
    remote = machine_with(rules=[("rsync", 23, "rsync: link_stat failed")])
    with pytest.raises(SystemExit, match="could not pin the source tree"):
        Snapshots("/work/projects").pin(remote, key="abc1234", image=mirrored("src"))


def test_a_host_that_dropped_while_pinning_reads_as_unreachable_not_as_a_broken_tree() -> None:
    """A transport blip is a fact about the host, and the dispatch retries rather than refuses."""
    remote = machine_with(rules=[("rsync", 255, "ssh: connect to host gold port 22: timed out")])
    with pytest.raises(HostUnreachable):
        Snapshots("/work/projects").pin(remote, key="abc1234", image=mirrored("src"))


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


@pins_on_this_host
def test_parallel_pins_freeze_the_listing_and_survive_mirror_replacement(
    sealed_mirror: tuple[Snapshots, Sealed, str],
) -> None:
    trees, image, digest = sealed_mirror
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [
            workers.submit(trees.pin, local, key="same", image=image, digest=digest)
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
    assert trees.pin(local, key="same", image=image, digest=digest) == str(frozen)
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
        trees.pin(local, key="wrong", image=image, digest=digest)
    assert not Path(trees.path("wrong")).exists()
    assert not list(Path(trees.base).glob(".pending.*"))


@pins_on_this_host
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("field", ["needs", "results"])
def test_live_paths_cannot_replace_a_verified_source_directory(
    sealed_mirror: tuple[Snapshots, Sealed, str],
    reuse: bool,
    field: str,
) -> None:
    trees, image, digest = sealed_mirror
    if reuse:
        trees.pin(local, key="overlap", image=image, digest=digest)
    kwargs = {"results": "./research/compression/pkg"} if field == "results" else {}
    if field == "needs":
        image = image.model_copy(update={"needs": ("./research/compression/pkg",)})
    with pytest.raises(SystemExit, match="live path overlaps source"):
        trees.pin(local, key="overlap", image=image, digest=digest, **kwargs)
    assert (Path(trees.root) / "research/compression/pkg/mod.py").read_text() == "v1\n"
    assert Path(trees.path("overlap")).exists() is reuse


@pins_on_this_host
def test_wrappers_are_frozen_by_bytes_and_not_repaired_after_corruption(
    sealed_mirror: tuple[Snapshots, Sealed, str],
) -> None:
    trees, image, digest = sealed_mirror
    root = Path(trees.root)
    frozen = Path(trees.pin(local, key="wrappers", image=image, digest=digest))
    payloads = (b"#!/bin/sh\r\n# non-UTF8: \xff\r\n", b"#!/bin/sh\n# second\n")
    for payload in payloads:
        staged = f".mainboard/dispatch/jobs/job-{sha256(payload).hexdigest()}.sh"
        (root / staged).write_bytes(payload)
        (root / staged).chmod(0o640)
        trees.pin(local, key="wrappers", image=image, digest=digest, script=staged)
        wrapper = frozen / Snapshots.script(staged)
        assert wrapper.read_bytes() == payload and wrapper.stat().st_mode & 0o777 == 0o640
        assert not wrapper.is_symlink() and not (frozen / WRAPPERS).is_symlink()
        (root / staged).write_text("changed mirror wrapper\n")
        trees.pin(local, key="wrappers", image=image, digest=digest, script=staged)
        assert wrapper.read_bytes() == payload
    assert len(list((frozen / WRAPPERS).glob("job-*.sh"))) == 2
    with pytest.raises(SystemExit, match="wrapper digest mismatch"):
        trees.pin(local, key="bad-wrapper", image=image, digest=digest, script=staged)
    assert not Path(trees.path("bad-wrapper")).exists()
    wrapper.write_text("corrupt frozen wrapper\n")
    with pytest.raises(SystemExit, match="wrapper digest mismatch"):
        trees.pin(local, key="wrappers", image=image, digest=digest, script=staged)
    assert wrapper.read_text() == "corrupt frozen wrapper\n"


@pins_on_this_host
def test_a_pinned_tree_survives_the_sync_that_rewrites_the_mirror_under_it(
    tmp_path: Path,
) -> None:
    """The fault itself, run for real: a mirror sync must not reach the code of a live job."""
    root = tmp_path / "projects"
    _mirror(root)
    pinned = Path(
        Snapshots(str(root)).pin(
            local,
            key="abc1234",
            image=mirrored(
                "research/compression",
                filters=("merge,- .gitignore", ":- .gitignore"),
                exclude=(".mainboard/",),
            ),
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

    later = tmp_path / "later"
    (later / "compression/pkg").mkdir(parents=True)
    (later / "compression/pkg/mod.py").write_text("v2\n", encoding="utf-8")
    # `-c` because the two files are seconds and bytes alike, which rsync's quick check reads
    # as unchanged; a real edit differs in one or the other.
    local["rsync"][["-ac", f"{later}/", f"{root}/research/"]]()
    assert (root / "research/compression/pkg/mod.py").read_text(encoding="utf-8") == "v2\n"
    assert frozen.read_text(encoding="utf-8") == "v1\n"


@pins_on_this_host
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
            local,
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
            local,
            key="abc1234-9f9f9f9f",
            image=Sealed(listing=listing, needs=("research/absent",)),
            digest=digest,
        )


@pins_on_this_host
def test_a_second_dispatch_of_one_tree_reuses_the_snapshot_instead_of_rebuilding_it(
    tmp_path: Path,
) -> None:
    """Thirty five jobs off one commit pay for one tree, and none of them waits for a rebuild."""
    root = tmp_path / "projects"
    _mirror(root)
    snapshots = Snapshots(str(root))
    image = mirrored("research/compression")
    pinned = Path(snapshots.pin(local, key="abc1234", image=image))
    (pinned / "research/compression/pkg/mod.py").unlink()
    assert snapshots.pin(local, key="abc1234", image=image) == str(pinned)
    assert not (pinned / "research/compression/pkg/mod.py").exists()


@pins_on_this_host
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
        snapshots.pin(local, key="abc1234", image=image, results="research/compression/raw")
    )
    second = Path(
        snapshots.pin(local, key="abc1234", image=image, results="research/compression/evidence")
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
    snapshots.pin(local, key="abc1234", image=image, results="research/compression/raw")
    assert first.joinpath("research/compression/raw").is_symlink()
    assert first.joinpath("research/compression/evidence").is_symlink()


@pins_on_this_host
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
        Snapshots(str(root)).pin(local, key="abc1234", image=mirrored("research/compression"))
    )
    generated = pinned / ".mainboard/envs/default/pixi.toml"
    mirrored_manifest = root / ".mainboard/envs/default/pixi.toml"
    replacement = generated.with_suffix(".toml.tmp")
    replacement.write_text("[workspace]\nname = 'pinned'\n", encoding="utf-8")
    replacement.replace(generated)
    assert generated.read_text(encoding="utf-8") != mirrored_manifest.read_text(encoding="utf-8")


def test_the_program_never_exits_since_a_login_shells_exit_runs_its_logout_file() -> None:
    """`exit` in a login shell runs `.bash_logout`, and a `clear_console` there fails without a
    terminal; under `set -e` that becomes the status of a pin that had nothing to do."""
    program = Snapshots("/mirror")._Snapshots__program(
        "/mirror/.mainboard/dispatch/sources/k",
        image=mirrored("a"),
        results="",
        prefix="",
        environment="default",
        stamp="k\n",
    )
    assert "exit" not in program
    assert program.startswith("set -euo pipefail\n")
    assert 'if [ ! -f "/mirror/.mainboard/dispatch/sources/k/' in program.replace(
        '"$mb_snap/', '"/mirror/.mainboard/dispatch/sources/k/'
    )
    assert program.endswith("\nfi")
