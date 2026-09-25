import os
import shutil
import subprocess
import time
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard import MissionError
from mainboard.dispatch.provenance import Source, SourceTree, Status, blob_of, listing, named
from mainboard.jobs.closure import Closure
from mainboard.jobs.target import Target

from ..support import Lab


def sealed(lab: Lab) -> tuple[Source, list]:
    target = Target.spelled([Lab.JOB], lab.root)
    assert target is not None
    closure = Closure.of(
        target,
        root=lab.root,
        distributions=Lab.DISTRIBUTIONS,
        environment=lab.root / Lab.ENVIRONMENT,
    )
    return SourceTree(lab.root).seal(closure.files, built=closure.built)


def test_source_capture_needs_neither_git_nor_a_repository(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "")
    assert not (lab.root / ".git").exists()
    captured, rows = sealed(lab)
    assert captured.identity == f"sha256:{captured.digest}"
    assert captured.key == f"sha256-{captured.digest}" and not captured.commit
    assert captured.digest == sha256(listing(rows).encode()).hexdigest()
    assert {row.status for row in rows} == {Status.SOURCE}
    assert len(blob_of(lab.root / Lab.JOB)) == 64


def test_changes_outside_the_closure_leave_its_identity_alone(lab: Lab) -> None:
    first, _ = sealed(lab)
    lab.write("research/other/scratch.py", "unrelated = True\n")
    lab.write("packages/sub/README.md", "not imported\n")
    assert sealed(lab)[0] == first


def test_edited_and_new_source_are_captured_without_committing(lab: Lab) -> None:
    first, _ = sealed(lab)
    lab.write("packages/sub/src/sub/thing.py", "THING = 2\n")
    second, _ = sealed(lab)
    assert second.digest != first.digest
    lab.write("research/camp/experiments/node/new.py", "NEW = 1\n")
    third, rows = sealed(lab)
    assert third.digest != second.digest
    assert any(row.path.endswith("/new.py") for row in rows)


def test_an_explicit_import_is_captured_even_if_ignored(lab: Lab) -> None:
    lab.write("research/camp/experiments/helper/x_generated.py", "X = 1\n")
    lab.write(Lab.JOB, "from ..helper import x_generated\napp = 1\n")
    first, rows = sealed(lab)
    assert any(row.path.endswith("x_generated.py") for row in rows)
    lab.write("research/camp/experiments/helper/x_generated.py", "X = 2\n")
    assert sealed(lab)[0].digest != first.digest


def test_built_extensions_are_hashed_as_actual_bytes(lab: Lab) -> None:
    path = "packages/core/src/core/native.so"
    file = lab.write(path, "stub")
    _, rows = SourceTree(lab.root).seal([path], built=[path])
    assert rows[0].status is Status.BUILT and rows[0].blob == blob_of(file)


def test_ignore_files_are_optional_and_nested_rules_apply(tmp_path: Path) -> None:
    lab = Lab(tmp_path)
    lab.write("src/keep.py", "x = 1")
    lab.write("src/drop.py", "x = 2")
    lab.write("src/nested/keep.py", "x = 3")
    lab.write("src/cache/x.py", "cache")
    lab.write("src/.gitignore", "drop.py\ncache/\n")
    lab.write(".gitignore", "src/nested/*.py\n")
    lab.write("src/nested/.gitignore", "!keep.py\n")
    lab.write("src/.env", "secret")
    lab.write("src/.git/config", "private")
    lab.write("src/__pycache__/x.pyc", "cache")
    files = SourceTree(tmp_path).kept("src")
    assert "src/keep.py" in files and "src/nested/keep.py" in files
    assert not set(files) & {"src/drop.py", "src/cache/x.py", "src/.env", "src/.git/config"}
    assert not any("__pycache__" in file for file in files)


def test_archives_preserve_original_bytes_and_reject_corruption(lab: Lab) -> None:
    tree = SourceTree(lab.root)
    first, rows = sealed(lab)
    manifest = listing(rows)
    archive = tree.archive(manifest)
    assert archive.name == f"{first.digest}.zip"
    original = (lab.root / Lab.JOB).read_bytes()
    lab.write(Lab.JOB, "edited after archival\n")
    assert tree.archive(manifest) == archive
    with ZipFile(archive) as stored:
        assert stored.read(Lab.JOB) == original
    with ZipFile(archive, "w") as stored:
        stored.writestr(".mainboard-source-listing.tsv", "wrong")
    with pytest.raises(MissionError, match="archive verification"):
        tree.archive(manifest)


@pytest.mark.parametrize(
    "tracked",
    [
        False,
        pytest.param(True, marks=pytest.mark.skipif(not shutil.which("git"), reason="no git")),
    ],
)
def test_sources_leave_out_the_data_a_host_still_ships(tmp_path: Path, tracked: bool) -> None:
    """Data, even what git tracks, is pinned by the trial that reads it, never archived; a host
    whose sync include names it still receives it."""
    lab = Lab(tmp_path)
    lab.write("experiments/law/test_law.py", "pass")
    lab.write("experiments/law/evidence/run.json", "{}")
    lab.write("datasets/experiments/law/rows.parquet", "rows")
    lab.write("corpora/text.txt", "words")
    lab.write("experiments/law/.card.lock", "")
    if tracked:
        subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    tree = SourceTree(tmp_path)
    assert tree.sources() == ["corpora/text.txt", "experiments/law/test_law.py"]
    lab.write("mainboard.toml", '[workspace]\nname = "w"\ndata = ["/corpora/"]\n')
    assert tree.sources() == [
        "datasets/experiments/law/rows.parquet",
        "experiments/law/evidence/run.json",
        "experiments/law/test_law.py",
        "mainboard.toml",
    ]
    shipped = tree.filter.files(["datasets", "experiments"])
    assert {"datasets/experiments/law/rows.parquet", "experiments/law/evidence/run.json"} <= set(
        shipped
    )


def test_a_killed_archival_leaves_one_partial_its_retry_replaces(lab: Lab) -> None:
    """The partial is named by its digest, and what a day has passed over is swept."""
    tree = SourceTree(lab.root)
    first, rows = sealed(lab)
    folder = lab.root / ".mainboard" / "source-archives"
    partial = folder / f"{first.digest}.zip.partial"
    lab.write(str(partial.relative_to(lab.root)), "killed mid-write")
    stale = [folder / "source-old" / "source.zip", folder / "other.zip.partial"]
    fresh = folder / "writing.zip.partial"
    for leftover in [*stale, fresh]:
        lab.write(str(leftover.relative_to(lab.root)), "left behind")
    day_ago = time.time() - 2 * 86_400
    for leftover in [stale[0].parent, stale[1]]:
        os.utime(leftover, (day_ago, day_ago))
    archive = tree.archive(listing(rows))
    with ZipFile(archive) as stored:
        assert stored.read(Lab.JOB) == (lab.root / Lab.JOB).read_bytes()
    assert sorted(path.name for path in folder.iterdir()) == [archive.name, fresh.name]


def test_source_change_before_archiving_is_rejected(lab: Lab) -> None:
    _, rows = sealed(lab)
    lab.write(Lab.JOB, "changed")
    with pytest.raises(MissionError, match="changed before source archival"):
        SourceTree(lab.root).archive(listing(rows))


@pytest.mark.parametrize("path", ["../outside", "/absolute", "bad\tpath", ".git/config", ".env"])
def test_private_or_escaping_paths_are_never_source(lab: Lab, path: str) -> None:
    with pytest.raises(MissionError):
        SourceTree(lab.root).seal([path])


def test_a_symlink_cannot_smuggle_external_bytes(lab: Lab, tmp_path: Path) -> None:
    outside = tmp_path / "secret"
    outside.write_text("private")
    (lab.root / "link").symlink_to(outside)
    with pytest.raises(MissionError, match="escapes"):
        SourceTree(lab.root).seal(["link"])


@given(identity=st.text(max_size=120))
def test_keys_cannot_escape_the_snapshot_directory(identity: str) -> None:
    key = named(identity)
    assert key and "/" not in key and not key.startswith(".") and len(key) <= 96
    assert named("..") == "source"
