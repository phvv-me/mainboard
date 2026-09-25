from pathlib import Path

import pytest

from mainboard.git import Outcome, Step

from .conftest import FOREIGN, OWNED, Workspace, posix_only

# A pre-receive hook that refuses `main` the way GitHub's branch protection does.
_PROTECT_MAIN = """#!/bin/sh
while read old new ref; do
  if [ "$ref" = "refs/heads/main" ]; then
    echo "remote: error: GH006: Protected branch update failed for refs/heads/main." >&2
    exit 1
  fi
done
"""


def _work(workspace: Workspace) -> None:
    """Commit a change in the library and let the tree record it, the usual state before a push."""
    (workspace.lib / "lib.txt").write_text("edited\n", encoding="utf-8")
    steps = workspace.tree().commit("Edit the library")
    assert all(step.outcome is Outcome.DONE for step in steps)


def _remote_head(workspace: Workspace, owner: str, name: str, ref: str = "main") -> str:
    return workspace.git(workspace.forge.bare(owner, name), "rev-parse", ref)


def test_push_goes_bottom_up_and_then_has_nothing_left_to_do(workspace: Workspace) -> None:
    _work(workspace)

    steps = workspace.tree().push()

    assert [(step.repo, step.outcome) for step in steps] == [
        ("packages/lib", Outcome.DONE),
        (".", Outcome.DONE),
    ]
    assert _remote_head(workspace, OWNED, "lib") == workspace.head(workspace.lib)
    assert _remote_head(workspace, "Pedrexus", "projects") == workspace.head(workspace.path)
    again = workspace.tree().push()
    assert [step.outcome for step in again] == [Outcome.CURRENT, Outcome.CURRENT]
    assert again[1].detail == "main level with origin/main"


def test_a_protected_main_gets_the_commit_on_a_branch_and_a_pull_request_asked_for(
    workspace: Workspace,
) -> None:
    _work(workspace)
    workspace.forge.hook(OWNED, "lib", _PROTECT_MAIN)

    lib, root = workspace.tree().push()

    assert lib.outcome is Outcome.DONE
    assert lib.detail.endswith("to mainboard/main, open a pull request")
    assert _remote_head(workspace, OWNED, "lib", "mainboard/main") == workspace.head(workspace.lib)
    assert root.outcome is Outcome.DONE
    waiting = workspace.tree().push()[0]
    assert waiting.outcome is Outcome.CURRENT
    assert waiting.detail.endswith("waits on mainboard/main")


@pytest.mark.parametrize(
    ("hook", "said"),
    [
        ("#!/bin/sh\necho 'remote: denied by policy' >&2\nexit 1\n", "denied by policy"),
        ("#!/bin/sh\necho 'remote: GH013: rule violation' >&2\nexit 1\n", "GH013: rule violation"),
    ],
)
def test_a_refused_push_fails_and_holds_every_parent(
    workspace: Workspace, hook: str, said: str
) -> None:
    """A refusal that is not protection fails; protection whose fallback is refused fails too."""
    _work(workspace)
    workspace.forge.hook(OWNED, "lib", hook)

    lib, root = workspace.tree().push()

    assert lib.outcome is Outcome.FAILED
    assert said in lib.detail
    assert root == Step(repo=".", outcome=Outcome.HELD, detail="packages/lib did not push")


def test_a_detached_head_is_pushed_only_when_a_remote_branch_already_holds_it(
    workspace: Workspace,
) -> None:
    lib, _ = workspace.tree().push()
    assert lib.outcome is Outcome.CURRENT
    assert lib.detail.endswith("already on origin/main")
    workspace.forge.commit(workspace.lib, "detached work", {"loose.txt": "loose\n"})

    lib, root = workspace.tree().push()

    assert lib.outcome is Outcome.HELD
    assert lib.detail.endswith("which no remote branch holds; commit first")
    assert root.outcome is Outcome.HELD


def test_a_parent_recording_a_foreign_commit_its_remote_lacks_is_held(
    workspace: Workspace,
) -> None:
    loose = workspace.forge.commit(workspace.ref, "local reference edit", {"x.txt": "x\n"})
    workspace.forge.commit(workspace.path, "Point at the edit", {})

    root = workspace.tree().push()[-1]

    assert root.outcome is Outcome.HELD
    assert root.detail == f"records references/ref@{loose[:7]}, which its remote does not hold"


def test_a_pointer_the_remote_gained_since_the_last_fetch_is_found_by_fetching(
    workspace: Workspace,
) -> None:
    upstream = workspace.forge.publish(workspace.forge.seed(FOREIGN, "ref"), {"up.txt": "up\n"})
    workspace.git(workspace.ref, "fetch", "-q", "origin")
    workspace.git(workspace.ref, "checkout", "-q", "--detach", upstream)
    workspace.git(workspace.ref, "update-ref", "-d", "refs/remotes/origin/main")
    workspace.forge.commit(workspace.path, "Follow the reference", {})

    root = workspace.tree().push()[-1]

    assert root.outcome is Outcome.DONE
    assert _remote_head(workspace, "Pedrexus", "projects") == workspace.head(workspace.path)


def test_a_diverged_branch_is_held_until_pulled(workspace: Workspace) -> None:
    colleague = workspace.colleague()
    colleague.forge.publish(colleague.path, {"theirs.txt": "theirs\n"})
    workspace.forge.commit(workspace.path, "mine", {"mine.txt": "mine\n"})
    workspace.git(workspace.path, "fetch", "-q", "origin")

    root = workspace.tree().push()[-1]

    assert root.outcome is Outcome.HELD
    assert root.detail == "diverged from origin/main: 1 ahead, 1 behind; pull first"


def test_a_branch_tracking_nothing_is_pushed_under_its_own_name_and_then_tracks_it(
    workspace: Workspace,
) -> None:
    workspace.git(workspace.lib, "switch", "-q", "-c", "feature")
    workspace.forge.commit(workspace.lib, "feature work", {"feature.txt": "f\n"})

    lib = workspace.tree().push()[0]

    assert lib.outcome is Outcome.DONE
    assert lib.detail.endswith("to origin/feature")
    assert workspace.git(workspace.lib, "rev-parse", "--abbrev-ref", "@{u}") == "origin/feature"


@posix_only
@pytest.mark.parametrize(("code", "outcome"), [("0", Outcome.DONE), ("1", Outcome.FAILED)])
def test_lfs_objects_are_uploaded_before_the_push_that_references_them(
    workspace: Workspace, fake_lfs: Path, code: str, outcome: Outcome
) -> None:
    (fake_lfs / "code").write_text(code, encoding="utf-8")
    (workspace.lib / ".gitattributes").write_text("*.bin filter=lfs\n", encoding="utf-8")
    _work(workspace)

    lib = workspace.tree().push()[0]

    assert lib.outcome is outcome
    assert "push origin main" in (fake_lfs / "calls").read_text(encoding="utf-8")
