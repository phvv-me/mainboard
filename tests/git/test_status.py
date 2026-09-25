from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mainboard.git import Tree
from mainboard.git.repo import Repo, owner_of, resolved
from mainboard.manifest.schema.git import GitPolicy

from .conftest import OWNED, Workspace

# A GitHub-shaped name: letters, digits, dots, dashes and underscores, never a separator.
_NAME = st.from_regex(r"[A-Za-z0-9][A-Za-z0-9._-]{0,20}", fullmatch=True)


@given(owner=_NAME, name=_NAME)
def test_every_remote_spelling_names_the_same_owner_and_one_without_an_owner_names_nobody(
    owner: str, name: str
) -> None:
    """https, scp-like ssh, ssh URLs and local paths on either platform all agree."""
    spellings = [
        f"https://github.com/{owner}/{name}.git",
        f"git@github.com:{owner}/{name}.git",
        f"ssh://git@github.com/{owner}/{name}",
        f"/srv/remotes/{owner}/{name}.git",
        f"C:\\remotes\\{owner}\\{name}.git",
    ]
    assert {owner_of(url) for url in spellings} == {owner}
    assert owner_of("") == owner_of(f"{name}.git") == ""


@given(owner=_NAME, other=_NAME, name=_NAME)
def test_a_relative_submodule_url_resolves_against_its_parents_remote(
    owner: str, other: str, name: str
) -> None:
    """`../x` is a sibling of the parent's repository, `../../o/x` another owner's."""
    base = f"https://github.com/{owner}/parent.git"
    assert owner_of(resolved(f"../{name}.git", base)) == owner
    assert owner_of(resolved(f"../../{other}/{name}.git", base)) == other
    assert resolved(f"git@github.com:{other}/{name}", base) == f"git@github.com:{other}/{name}"


@pytest.mark.parametrize(
    ("owner", "owned"),
    [("Pedrexus", True), ("PHVV-ME", True), ("other", False), ("", False)],
)
def test_ownership_is_the_roots_owner_or_a_declared_one_in_any_case(
    owner: str, owned: bool
) -> None:
    assert GitPolicy(owners=[OWNED]).owns(owner, "pedrexus") is owned


def test_the_policy_spells_its_patterns_as_pathspecs_and_its_ceiling_in_bytes() -> None:
    policy = GitPolicy.model_validate({"ceiling-mb": 2, "never-commit": ["a/**"]})
    assert policy.ceiling_bytes == 2 << 20
    assert policy.outside == [":(exclude,glob)a/**"]
    assert policy.inside == [":(glob)a/**"]


def test_status_reads_only_owned_repositories_and_stops_at_foreign_ones(
    workspace: Workspace,
) -> None:
    """The walk never enters `references/ref` nor the `vendor/dep` under the owned library."""
    root, lib = workspace.tree().status()
    assert (root.repo, root.owner, root.branch, root.upstream) == (
        ".",
        "Pedrexus",
        "main",
        "origin/main",
    )
    assert root.published == "origin/main"
    assert (lib.repo, lib.owner, lib.branch, lib.upstream) == (
        "packages/lib",
        OWNED,
        "detached",
        "origin/main",
    )
    assert (lib.ahead, lib.behind, lib.changed, lib.untracked) == (0, 0, 0, 0)


def test_status_counts_what_the_next_commit_would_take(workspace: Workspace) -> None:
    """An evidence artifact is neither changed nor untracked, since no commit will take it."""
    (workspace.lib / "lib.txt").write_text("edited\n", encoding="utf-8")
    (workspace.lib / "notes.md").write_text("new\n", encoding="utf-8")
    artifact = workspace.lib / "run" / "evidence" / "artifacts" / "out.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"x")
    workspace.forge.commit(workspace.lib, "local", {})
    lib = workspace.tree().status()[1]
    assert (lib.changed, lib.untracked, lib.ahead) == (0, 0, 1)
    assert lib.published == ""
    (workspace.lib / "lib.txt").write_text("again\n", encoding="utf-8")
    (workspace.lib / "more.md").write_text("new\n", encoding="utf-8")
    lib = workspace.tree().status()[1]
    assert (lib.changed, lib.untracked) == (1, 1)


def test_a_head_with_nothing_to_track_counts_against_nothing(workspace: Workspace) -> None:
    """A branch tracking no upstream, then a detached HEAD whose trunk the remote lacks."""
    workspace.git(workspace.lib, "switch", "-q", "-c", "feature")
    lib = workspace.tree().status()[1]
    assert (lib.branch, lib.upstream, lib.ahead, lib.behind) == ("feature", "", 0, 0)
    assert lib.published == "origin/main"
    workspace.git(workspace.lib, "switch", "-q", "--detach")
    workspace.git(workspace.lib, "update-ref", "-d", "refs/remotes/origin/main")
    lib = workspace.tree().status()[1]
    assert (lib.upstream, lib.published) == ("", "")


def test_the_trunk_is_declared_then_the_remote_head_then_main(workspace: Workspace) -> None:
    tree = workspace.tree()
    lib, ref = tree.root.children
    assert lib.trunk() == "main"
    workspace.git(workspace.forge.seed("other", "ref"), "push", "-q", "origin", "main:trunk")
    workspace.git(workspace.ref, "fetch", "-q", "origin")
    workspace.git(workspace.ref, "remote", "set-head", "origin", "trunk")
    assert ref.trunk() == "trunk"
    workspace.git(workspace.ref, "remote", "set-head", "origin", "-d")
    assert ref.trunk() == "main"


def test_a_repository_with_no_origin_is_owned_by_nobody(tmp_path: Path) -> None:
    """A root with no remote owns nothing, so no verb has anything to walk."""
    Repo(tmp_path, owns=bool).git.out("init", "-q")
    tree = Tree(tmp_path, GitPolicy(owners=[OWNED]))
    assert tree.root.url == ""
    assert tree.status() == []
    assert tree.root.pointers() == {}
    assert tree.root.homes("HEAD") == []


def test_an_unchecked_out_submodule_answers_with_its_declared_url(workspace: Workspace) -> None:
    workspace.git(workspace.path, "submodule", "deinit", "-q", "-f", "references/ref")
    ref = workspace.tree().root.children[1]
    assert not ref.initialized
    assert ref.url.endswith("ref.git")
    assert ref.owner == "other"


def test_lfs_is_read_off_the_attributes_file(workspace: Workspace) -> None:
    lib = workspace.tree().root.children[0]
    assert not lib.lfs()
    (workspace.lib / ".gitattributes").write_text("*.txt text\n", encoding="utf-8")
    assert not lib.lfs()
    (workspace.lib / ".gitattributes").write_text("*.bin filter=lfs\n", encoding="utf-8")
    assert lib.lfs()
