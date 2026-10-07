"""The tree verbs on the states a real workspace drifts into.

A nested submodule checkout whose `.git` names a directory that is gone aborts a bare `git
status` in every repository above it (research/llm's googletest, 2026-09-30). `mb git status` and
`mb git check` never recurse, so they survive it, and they are the ones that name the path.

The center moving to Windows (2026-10-01) brought the rest: links checked out as plain files, a
workspace behind an upstream another machine pushed to, and a repository nested in the tree with
no `.gitmodules` entry. `mb git commit` settles each of them rather than stopping on it.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(where: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in `where`, local-path submodules allowed and an author named."""
    settings = ("protocol.file.allow=always", "user.name=it", "user.email=it@example.invalid")
    flags = [word for setting in settings for word in ("-c", setting)]
    return subprocess.run(
        ["git", "-C", str(where), *flags, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def committed(where: Path) -> Path:
    """A repository at `where` holding one commit."""
    where.mkdir(parents=True)
    git(where, "init", "--quiet", "--initial-branch=main")
    (where / "README").write_text("one\n", encoding="utf-8", newline="\n")
    git(where, "add", "README")
    git(where, "commit", "--quiet", "-m", "one")
    return where


@pytest.fixture
def broken(workspace: Path, tmp_path_factory) -> Path:
    """The workspace as a repository whose foreign submodule holds an unreadable nested one."""
    remotes = tmp_path_factory.mktemp("remotes")
    nested = committed(remotes / "other" / "nested")
    library = committed(remotes / "other" / "library")
    git(library, "submodule", "add", "--quiet", nested.as_posix(), "deps/nested")
    git(library, "commit", "--quiet", "-m", "nest")
    git(workspace, "init", "--quiet", "--initial-branch=main")
    git(workspace, "config", "user.name", "it")
    git(workspace, "config", "user.email", "it@example.invalid")
    git(workspace, "remote", "add", "origin", (remotes / "me" / "root.git").as_posix())
    git(workspace, "submodule", "add", "--quiet", library.as_posix(), "vendor/library")
    git(workspace / "vendor" / "library", "submodule", "update", "--init", "--quiet")
    git(workspace, "add", "-A")
    git(workspace, "commit", "--quiet", "-m", "vendor")
    assert git(workspace, "status", "--porcelain").returncode == 0
    pointer = workspace / "vendor" / "library" / "deps" / "nested" / ".git"
    # Replaced rather than rewritten: git hides the file on Windows, which refuses a write.
    pointer.unlink()
    pointer.write_text("gitdir: ../../../../.git/modules/gone\n", encoding="utf-8", newline="\n")
    assert git(workspace, "status", "--porcelain").returncode != 0
    return workspace


@pytest.fixture
def tracked(workspace: Path, tmp_path_factory) -> Path:
    """The workspace as a repository with an author, its manifest committed and pushed to a bare
    `origin`, the remote that makes it the workspace's own."""
    remote = tmp_path_factory.mktemp("remotes") / "root.git"
    git(workspace, "init", "--quiet", "--bare", "--initial-branch=main", remote.as_posix())
    git(workspace, "init", "--quiet", "--initial-branch=main")
    git(workspace, "config", "user.name", "it")
    git(workspace, "config", "user.email", "it@example.invalid")
    git(workspace, "remote", "add", "origin", remote.as_posix())
    git(workspace, "add", "-A")
    git(workspace, "commit", "--quiet", "-m", "manifest")
    git(workspace, "push", "--quiet", "-u", "origin", "main")
    return workspace


def steps(ran) -> dict[str, dict[str, str]]:
    """The rows of a verb's `--json` answer, keyed by repository."""
    return {row["repo"]: row for row in json.loads(ran.out)}


def test_status_survives_an_unreadable_checkout_and_names_it(mb, broken: Path) -> None:
    ran = mb("git", "status", "--json")
    assert ran.code == 0, ran.said
    (root,) = json.loads(ran.out)
    assert root["repo"] == "." and "deps/nested" in root["broken"]


def test_check_survives_an_unreadable_checkout_and_names_it(mb, broken: Path) -> None:
    ran = mb("git", "check", "--json", timeout=300)
    found = [row for row in json.loads(ran.out) if row["check"] == "submodule"]
    assert [row["repo"] for row in found] == ["."]
    detail = found[0]["detail"]
    assert "deps/nested" in detail and "git submodule update --init" in detail


@pytest.mark.parametrize("makes_links", ["true", "false"])
def test_commit_puts_back_links_written_as_files_and_withholds_an_edited_one(
    mb, tracked: Path, makes_links: str
) -> None:
    git(tracked, "config", "core.symlinks", makes_links)
    (tracked / "target.txt").write_text("kept\n", encoding="utf-8", newline="\n")
    links = ("copied", "spelled", "edited")
    try:
        for name in links:
            (tracked / name).symlink_to("target.txt")
    except OSError:
        pytest.skip("this machine cannot make symbolic links")
    git(tracked, "add", "-A")
    git(tracked, "commit", "--quiet", "-m", "links")
    head = git(tracked, "rev-parse", "HEAD").stdout
    for name in links:
        (tracked / name).unlink()
    (tracked / "copied").write_bytes((tracked / "target.txt").read_bytes())
    (tracked / "spelled").write_text("target.txt", encoding="utf-8", newline="")
    (tracked / "edited").write_text("changed through the copy\n", encoding="utf-8")
    ran = mb("git", "commit", "-m", "nothing to take", "--all", "--json")
    detail = steps(ran)["."]["detail"]
    # Without links git reads `spelled` as the link itself, so only the copy needs putting back.
    relinked = "copied, spelled" if makes_links == "true" else "copied"
    assert f"relinked {relinked};" in detail and "withheld edited" in detail
    assert not git(tracked, "status", "--porcelain", "--", "copied", "spelled").stdout
    assert git(tracked, "rev-parse", "HEAD").stdout == head


@pytest.mark.parametrize("verb", ["commit", "pull"])
@pytest.mark.parametrize("edited", ["mine.txt", "shared.txt"])
def test_a_diverged_workspace_merges_its_upstream_or_names_the_conflict(
    mb, tracked: Path, tmp_path_factory, verb: str, edited: str
) -> None:
    remote = git(tracked, "remote", "get-url", "origin").stdout.strip()
    other = tmp_path_factory.mktemp("other") / "root"
    git(tracked, "clone", "--quiet", remote, other.as_posix())
    (other / "shared.txt").write_text("theirs\n", encoding="utf-8", newline="\n")
    git(other, "add", "-A")
    git(other, "commit", "--quiet", "-m", "theirs")
    git(other, "push", "--quiet")
    git(tracked, "fetch", "--quiet")
    (tracked / edited).write_text("mine\n", encoding="utf-8", newline="\n")
    if verb == "pull":
        git(tracked, "add", "-A")
        git(tracked, "commit", "--quiet", "-m", "mine")
    root = steps(
        mb("git", verb, "--json", *(("-m", "mine", "--all") if verb == "commit" else ()))
    )["."]
    assert not (tracked / ".git" / "MERGE_HEAD").exists()
    if edited == "shared.txt":
        assert "conflicts in shared.txt" in root["detail"]
        assert root["outcome"] == ("done" if verb == "commit" else "held")
    else:
        assert "merged 1 from origin/main" in root["detail"]
        assert (tracked / "shared.txt").read_text(encoding="utf-8") == "theirs\n"
    assert "mine" in git(tracked, "log", "--format=%s").stdout.split("\n")


def test_commit_withholds_a_nested_repository_gitmodules_does_not_declare(
    mb, tracked: Path
) -> None:
    committed(tracked / "nested")
    (tracked / "plain.txt").write_text("plain\n", encoding="utf-8", newline="\n")
    root = steps(mb("git", "commit", "-m", "plain", "--all", "--json"))["."]
    assert root["outcome"] == "done" and "withheld nested/" in root["detail"]
    listed = git(tracked, "ls-tree", "--name-only", "HEAD").stdout.split()
    assert "plain.txt" in listed and "nested" not in listed


def test_commit_takes_only_the_named_paths_and_leaves_the_index_alone(mb, tracked: Path) -> None:
    for name in ("mine.txt", "theirs.txt", "staged.txt"):
        (tracked / name).write_text(f"{name}\n", encoding="utf-8", newline="\n")
    git(tracked, "add", "staged.txt")
    root = steps(mb("git", "commit", "-m", "mine", "mine.txt", "--json"))["."]
    assert root["outcome"] == "done", root
    assert git(tracked, "show", "--name-only", "--format=", "HEAD").stdout.split() == ["mine.txt"]
    status = git(tracked, "status", "--porcelain").stdout.splitlines()
    assert "A  staged.txt" in status and "?? theirs.txt" in status


def test_commit_takes_a_named_file_that_replaced_a_link_and_refuses_a_flattened_one(
    mb, tracked: Path
) -> None:
    git(tracked, "config", "core.symlinks", "true")
    (tracked / "target.txt").write_text("kept\n", encoding="utf-8", newline="\n")
    try:
        for name in ("rendered", "flattened"):
            (tracked / name).symlink_to("target.txt")
    except OSError:
        pytest.skip("this machine cannot make symbolic links")
    git(tracked, "add", "-A")
    git(tracked, "commit", "--quiet", "-m", "links")
    for name in ("rendered", "flattened"):
        (tracked / name).unlink()
    (tracked / "rendered").write_text("rendered on purpose\n", encoding="utf-8", newline="\n")
    (tracked / "flattened").write_text("target.txt", encoding="utf-8", newline="")

    refused = steps(mb("git", "commit", "-m", "flat", "flattened", "--json"))["."]
    taken = steps(mb("git", "commit", "-m", "render", "rendered", "--json"))["."]

    assert refused["outcome"] == "held" and "refused flattened" in refused["detail"]
    assert taken["outcome"] == "done", taken
    assert git(tracked, "ls-files", "-s", "rendered").stdout.startswith("100644")


def test_commit_without_paths_commits_the_index_and_refuses_an_empty_one(
    mb, tracked: Path
) -> None:
    (tracked / "loose.txt").write_text("loose\n", encoding="utf-8", newline="\n")
    refused = mb("git", "commit", "-m", "nothing")
    assert refused.code != 0 and "--all" in refused.said
    (tracked / "staged.txt").write_text("staged\n", encoding="utf-8", newline="\n")
    git(tracked, "add", "staged.txt")
    root = steps(mb("git", "commit", "-m", "staged", "--json"))["."]
    assert root["outcome"] == "done", root
    assert git(tracked, "show", "--name-only", "--format=", "HEAD").stdout.split() == [
        "staged.txt"
    ]
    assert "?? loose.txt" in git(tracked, "status", "--porcelain").stdout


def test_commit_refuses_a_named_never_commit_path(mb, tracked: Path) -> None:
    data = tracked / "datasets" / "rows.csv"
    data.parent.mkdir()
    data.write_text("a,b\n", encoding="utf-8", newline="\n")
    head = git(tracked, "rev-parse", "HEAD").stdout
    root = steps(mb("git", "commit", "-m", "data", "datasets/rows.csv", "--json"))["."]
    assert root["outcome"] == "held" and "refused datasets/rows.csv" in root["detail"]
    assert git(tracked, "rev-parse", "HEAD").stdout == head


def test_commit_records_the_pointer_of_a_submodule_that_committed(mb, tracked: Path) -> None:
    # Beside the root's own remote, so its URL names the same owner and the tree owns it.
    remote = Path(git(tracked, "remote", "get-url", "origin").stdout.strip())
    library = committed(remote.parent / "library")
    git(tracked, "submodule", "add", "--quiet", library.as_posix(), "library")
    git(tracked, "commit", "--quiet", "-m", "library")
    checkout = tracked / "library"
    git(checkout, "config", "user.name", "it")
    git(checkout, "config", "user.email", "it@example.invalid")
    (checkout / "change.txt").write_text("change\n", encoding="utf-8", newline="\n")
    (tracked / "unrelated.txt").write_text("unrelated\n", encoding="utf-8", newline="\n")
    ran = steps(mb("git", "commit", "-m", "change", "library/change.txt", "--json"))
    assert ran["library"]["outcome"] == "done" and ran["."]["outcome"] == "done", ran
    assert git(tracked, "show", "--name-only", "--format=", "HEAD").stdout.split() == ["library"]
    assert "?? unrelated.txt" in git(tracked, "status", "--porcelain").stdout
